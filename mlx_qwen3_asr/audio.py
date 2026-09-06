"""Audio loading and mel spectrogram computation for Qwen3-ASR.

Handles audio I/O via ffmpeg subprocess and computes Whisper-compatible
log-mel spectrograms. The default feature path is a native custom mel
implementation.

Key parameters:
    - Sample rate: 16000 Hz
    - FFT size: 400 (25ms window at 16kHz)
    - Hop length: 160 (10ms stride at 16kHz)
    - Mel bins: 128 (Slaney-normalized)
    - No forced 30s truncation in compute_features()
"""

from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Union

import mlx.core as mx
import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
NUM_MEL_BINS = 128
WHISPER_MAX_FRAMES = 3000  # 30s at 10ms hop


AudioSource = Union[str, Path, np.ndarray, Tuple[np.ndarray, int]]


def _ffmpeg_missing_message() -> str:
    if sys.platform == "darwin":
        install = "brew install ffmpeg"
    elif sys.platform.startswith("linux"):
        install = "sudo apt-get update && sudo apt-get install -y ffmpeg"
    elif sys.platform.startswith("win"):
        install = "winget install Gyan.FFmpeg"
    else:
        install = "Install ffmpeg and ensure it is available on PATH."
    return f"ffmpeg not found on PATH. Install and retry: {install}"


def load_audio_np(
    source: AudioSource,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Load audio as mono float32 numpy array at the target sample rate.

    Args:
        source: One of:
            - str or Path: path to an audio file (any format ffmpeg supports)
            - np.ndarray: raw waveform assumed to be at target sample rate
            - tuple of (np.ndarray, int): raw waveform with its sample rate
        sr: Target sample rate. Default 16000.

    Returns:
        Mono float32 numpy array of shape (n_samples,).

    Raises:
        RuntimeError: If ffmpeg fails or is not installed.
        ValueError: If source type is not recognized.
    """
    if isinstance(source, tuple):
        audio_np, orig_sr = source
        audio_np = _sanitize_audio_array(audio_np)
        if orig_sr != sr:
            audio_np = _resample_via_ffmpeg(audio_np, orig_sr, sr)
        return np.asarray(audio_np, dtype=np.float32)

    if isinstance(source, np.ndarray):
        audio_np = _sanitize_audio_array(source)
        return np.asarray(audio_np, dtype=np.float32)

    if isinstance(source, (str, Path)):
        return _load_audio_file(str(source), sr)

    raise ValueError(
        f"Unsupported source type: {type(source)}. "
        "Expected str, Path, np.ndarray, or (np.ndarray, int)."
    )


def load_audio(
    source: AudioSource,
    sr: int = SAMPLE_RATE,
) -> mx.array:
    """Backward-compatible wrapper that returns an MLX array."""
    return mx.array(load_audio_np(source, sr=sr))


def _sanitize_audio_array(source: np.ndarray) -> np.ndarray:
    """Canonicalize in-memory audio arrays to mono float32 in [-1, 1]."""
    audio_np = np.asarray(source)
    if audio_np.size == 0:
        return np.array([], dtype=np.float32)
    if audio_np.ndim > 2:
        raise ValueError(
            f"Audio arrays must be 1-D or 2-D, got shape {audio_np.shape}"
        )

    # Normalize integer PCM to match file decode semantics.
    if np.issubdtype(audio_np.dtype, np.integer):
        audio_np = _normalize_integer_pcm(audio_np)
    else:
        audio_np = audio_np.astype(np.float32, copy=False)

    if audio_np.ndim == 2:
        n0, n1 = int(audio_np.shape[0]), int(audio_np.shape[1])
        if n0 <= 8 and n1 <= 8:
            if n0 == n1:
                channel_axis = 1
            else:
                channel_axis = 0 if n0 < n1 else 1
        elif n0 <= 8:
            channel_axis = 0
        elif n1 <= 8:
            channel_axis = 1
        else:
            channel_axis = 1
        audio_np = audio_np.mean(axis=channel_axis)

    return np.asarray(audio_np, dtype=np.float32)


def _normalize_integer_pcm(audio_np: np.ndarray) -> np.ndarray:
    """Convert integer PCM arrays to float32 in approximately [-1, 1]."""
    info = np.iinfo(audio_np.dtype)
    x: np.ndarray = audio_np.astype(np.float32)
    if info.min >= 0:
        midpoint = (info.max + 1) / 2.0
        return (x - midpoint) / midpoint
    scale = float(max(abs(info.min), info.max))
    return x / scale


def _load_audio_file(path: str, sr: int) -> np.ndarray:
    """Load audio file using ffmpeg, returning mono float32 at target sample rate."""
    wav_audio = _try_load_wav_fast(path, sr)
    if wav_audio is not None:
        return wav_audio.astype(np.float32, copy=False)

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    except FileNotFoundError:
        raise RuntimeError(_ffmpeg_missing_message())

    return np.frombuffer(result.stdout, np.int16).astype(np.float32) / 32768.0


def _try_load_wav_fast(path: str, target_sr: int) -> np.ndarray | None:
    """Fast-path WAV loader using stdlib for uncompressed PCM WAV files.

    Returns None when the file is not a supported WAV variant so callers can
    fall back to ffmpeg.
    """
    if not str(path).lower().endswith(".wav"):
        return None

    try:
        data = Path(path).read_bytes()
    except OSError:
        return None

    parsed = _parse_wav_bytes(data)
    if parsed is None:
        return None

    audio, orig_sr = parsed
    if audio.size == 0:
        return np.array([], dtype=np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if orig_sr != target_sr:
        audio = _resample_via_ffmpeg(audio.astype(np.float32), orig_sr, target_sr)

    return audio.astype(np.float32)


def _parse_wav_bytes(data: bytes) -> tuple[np.ndarray, int] | None:
    """Parse a RIFF/WAVE byte stream.

    Supports little-endian PCM (format 1) and IEEE float (format 3).
    Returns decoded samples and sample rate, or None if unsupported.
    """
    if len(data) < 12:
        return None
    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None

    fmt_chunk: bytes | None = None
    data_chunk: bytes | None = None
    i = 12
    while i + 8 <= len(data):
        chunk_id = data[i : i + 4]
        chunk_size = int.from_bytes(data[i + 4 : i + 8], "little")
        start = i + 8
        end = start + chunk_size
        if end > len(data):
            return None
        chunk = data[start:end]
        if chunk_id == b"fmt ":
            fmt_chunk = chunk
        elif chunk_id == b"data":
            data_chunk = chunk
        i = end + (chunk_size & 1)

    if fmt_chunk is None or data_chunk is None or len(fmt_chunk) < 16:
        return None

    audio_format = int.from_bytes(fmt_chunk[0:2], "little")
    n_channels = int.from_bytes(fmt_chunk[2:4], "little")
    sample_rate = int.from_bytes(fmt_chunk[4:8], "little")
    bits_per_sample = int.from_bytes(fmt_chunk[14:16], "little")
    if n_channels <= 0 or sample_rate <= 0:
        return None

    if audio_format == 1:  # PCM
        if bits_per_sample % 8 != 0:
            return None
        sample_width = bits_per_sample // 8
        decoded = _decode_pcm_bytes(data_chunk, sample_width)
        if decoded is None:
            return None
    elif audio_format == 3:  # IEEE float
        if bits_per_sample == 32:
            decoded = np.frombuffer(data_chunk, dtype="<f4").astype(np.float32)
        elif bits_per_sample == 64:
            decoded = np.frombuffer(data_chunk, dtype="<f8").astype(np.float32)
        else:
            return None
    else:
        return None

    n = (decoded.size // n_channels) * n_channels
    decoded = decoded[:n]
    if n_channels > 1:
        decoded = decoded.reshape(-1, n_channels)
    return decoded, sample_rate


def _decode_pcm_bytes(raw: bytes, sample_width: int) -> np.ndarray | None:
    """Decode PCM bytes from WAV data to float32 in [-1, 1]."""
    if sample_width == 1:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (x - 128.0) / 128.0

    if sample_width == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        return x / 32768.0

    if sample_width == 3:
        b = np.frombuffer(raw, dtype=np.uint8)
        if len(b) % 3 != 0:
            return None
        b = b.reshape(-1, 3).astype(np.int32)
        x = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        sign_mask = 1 << 23
        x = np.where((x & sign_mask) != 0, x - (1 << 24), x)
        return x.astype(np.float32) / float(1 << 23)

    if sample_width == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        return x / float(1 << 31)

    return None


def _resample_via_ffmpeg(
    audio: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    """Resample audio array by piping through ffmpeg."""
    # Convert to s16le bytes
    audio_s16 = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
    input_bytes = audio_s16.tobytes()

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(orig_sr),
        "-i",
        "-",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(target_sr),
        "-",
    ]
    try:
        result = subprocess.run(
            cmd, input=input_bytes, capture_output=True, check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to resample audio: {e.stderr.decode()}") from e
    except FileNotFoundError:
        raise RuntimeError(_ffmpeg_missing_message())

    return np.frombuffer(result.stdout, np.int16).astype(np.float32) / 32768.0


def mel_filters(n_mels: int = NUM_MEL_BINS) -> mx.array:
    """Load pre-computed mel filterbank from assets/mel_filters.npz.

    Args:
        n_mels: Number of mel bins. Must match a key in the .npz file.

    Returns:
        Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1).

    Raises:
        FileNotFoundError: If mel_filters.npz is missing.
        KeyError: If the requested n_mels key is not in the file.
    """
    return mx.array(_mel_filters_np(n_mels))


@lru_cache(maxsize=4)
def _mel_filters_np(n_mels: int) -> np.ndarray:
    """Load and cache mel filterbank as numpy array by mel-bin count."""
    assets_dir = Path(__file__).parent / "assets"
    npz_path = assets_dir / "mel_filters.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Mel filterbank file not found: {npz_path}. "
            "Run: python scripts/generate_mel_filters.py"
        )
    filterbank = np.load(npz_path)
    key = f"mel_{n_mels}"
    if key not in filterbank:
        raise KeyError(
            f"Key '{key}' not found in {npz_path}. "
            f"Available keys: {list(filterbank.keys())}"
        )
    return np.array(filterbank[key], dtype=np.float32, copy=False)


@lru_cache(maxsize=2)
def _hann_window(n_fft: int) -> mx.array:
    """Create and cache Hann window used by STFT."""
    return mx.array(np.hanning(n_fft + 1)[:-1].astype(np.float32))


def compute_features(
    audio_np: np.ndarray,
    sr: int = SAMPLE_RATE,
    padding: str = "do_not_pad",
) -> tuple[mx.array, mx.array]:
    """Compute mel spectrogram features.

    Native custom mel path is always used.

    Args:
        audio_np: Raw waveform as numpy array, shape (n_samples,).
        sr: Sample rate. Default 16000.
        padding: Feature padding mode.
            `"do_not_pad"` and `"longest"` keep the natural frame length.
            `"max_length"` pads short clips to 3000 frames while preserving
            true `feature_lens`.

    Returns:
        Tuple of (mel_features, feature_lens):
            mel_features: shape (1, 128, n_frames) as mx.array
            feature_lens: shape (1,) with actual frame count as mx.array
    """
    audio_np = np.asarray(audio_np, dtype=np.float32)

    if sr != SAMPLE_RATE:
        audio_np = _resample_via_ffmpeg(audio_np, sr, SAMPLE_RATE)

    mode = str(padding).strip().lower()
    if mode not in {"do_not_pad", "max_length", "longest"}:
        raise ValueError(
            f"Unsupported padding mode '{padding}'. "
            "Expected one of: do_not_pad, max_length, longest."
        )

    mel = log_mel_spectrogram(mx.array(audio_np))
    actual_frames = int(mel.shape[-1])
    if mode == "max_length" and actual_frames < WHISPER_MAX_FRAMES:
        pad = WHISPER_MAX_FRAMES - actual_frames
        mel = mx.concatenate(
            [mel, mx.zeros((int(mel.shape[0]), pad), dtype=mel.dtype)],
            axis=1,
        )
    mel_mx = mel[None, :, :].astype(mx.float32)
    feature_lens = mx.array([actual_frames], dtype=mx.int32)
    return mel_mx, feature_lens


def _reflect_pad(x: mx.array, pad_len: int) -> mx.array:
    """Apply reflect padding to a 1-D array.

    Mimics np.pad(x, pad_len, mode='reflect'). For an input [a, b, c, d]
    with pad_len=2, the result is [c, b, a, b, c, d, c, b].

    Args:
        x: 1-D input array.
        pad_len: Number of samples to pad on each side.

    Returns:
        Padded 1-D array of length len(x) + 2 * pad_len.
    """
    if pad_len == 0:
        return x
    # Left pad: reversed slice x[1 : pad_len + 1]
    left = x[1 : pad_len + 1][::-1]
    # Right pad: reversed slice x[-(pad_len + 1) : -1]
    right = x[-(pad_len + 1) : -1][::-1]
    return mx.concatenate([left, x, right])


def stft(
    x: mx.array,
    window: mx.array,
    nperseg: int = N_FFT,
    noverlap: int | None = None,
) -> mx.array:
    """Compute Short-Time Fourier Transform using MLX.

    Uses mx.as_strided for efficient frame extraction and mx.fft.rfft for the
    FFT computation. Applies reflect padding of nperseg // 2 on each side,
    matching the Whisper/librosa convention.

    Args:
        x: Input waveform, shape (n_samples,).
        window: Window function, shape (nperseg,).
        nperseg: Segment length (FFT size). Default 400.
        noverlap: Overlap between segments. Default nperseg - HOP_LENGTH.

    Returns:
        Complex STFT matrix, shape (n_frames, n_fft // 2 + 1).
    """
    if noverlap is None:
        noverlap = nperseg - HOP_LENGTH

    hop = nperseg - noverlap
    pad_len = nperseg // 2
    if x.shape[0] <= pad_len:
        # Reflect padding needs more samples than the pad width, and newer MLX
        # raises an opaque as_strided error instead of producing zero frames.
        raise ValueError(
            "Audio too short for STFT: "
            f"{int(x.shape[0])} samples, need more than {pad_len}."
        )

    # Reflect padding (mx.pad does not support mode="reflect", so do it manually)
    x = _reflect_pad(x, pad_len)

    # Frame extraction via as_strided
    # mx.as_strided uses element-based strides (not byte-based like numpy)
    num_frames = 1 + (x.shape[0] - nperseg) // hop
    frames = mx.as_strided(x, shape=(num_frames, nperseg), strides=(hop, 1))

    # Apply window and compute FFT
    windowed = frames * window
    return mx.fft.rfft(windowed)


def log_mel_spectrogram(
    audio: mx.array,
    n_mels: int = NUM_MEL_BINS,
) -> mx.array:
    """Compute log-mel spectrogram from raw audio waveform.

    Follows the Whisper-style pipeline:
    1. STFT with Hann window (400-sample FFT, 160-sample hop)
    2. Power spectrogram (magnitude squared)
    3. Mel filterbank projection
    4. Log10 with clamping and normalization

    Args:
        audio: Raw waveform, shape (n_samples,). Must be float32.
        n_mels: Number of mel bins (default 128).

    Returns:
        Log-mel spectrogram, shape (n_mels, n_frames). No zero-padding to
        a fixed length; the caller handles padding if needed.

    Raises:
        ValueError: If audio is empty or too short to produce one mel frame.
    """
    if audio.size == 0:
        raise ValueError("Cannot compute mel spectrogram of empty audio.")

    # Reuse cached Hann window to avoid per-call host allocation.
    window = _hann_window(N_FFT)

    # STFT
    freqs = stft(audio, window, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)
    magnitudes = mx.abs(freqs) ** 2  # (n_frames, n_fft // 2 + 1)

    # Mel filterbank projection
    filters = mel_filters(n_mels)  # (n_mels, n_fft // 2 + 1)
    mel_spec = filters @ magnitudes.T  # (n_mels, n_frames)
    n_frames = int(mel_spec.shape[1])
    if n_frames <= 1:
        raise ValueError(
            "Audio too short for log-mel extraction after Whisper frame trim: "
            f"{int(audio.size)} samples produced {n_frames} STFT frame(s)."
        )

    # Whisper feature extraction trims the final STFT frame before clamping.
    # This makes output length exactly len(audio) // hop_length for do_not_pad.
    log_spec = mx.log10(mx.maximum(mel_spec, 1e-10))[:, :-1]

    # Log scale with clamping and normalization
    log_spec = mx.maximum(log_spec, mx.max(log_spec) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec
