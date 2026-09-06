"""Dataclass configs for Qwen3-ASR model components.

Pure Python dataclasses with no MLX imports. Supports both 0.6B and 1.7B
model sizes via class methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Public default model IDs (kept centralized to avoid drift across CLI/API).
DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
ACCURACY_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"


@dataclass
class AudioEncoderConfig:
    """Configuration for the Qwen3-ASR audio encoder.

    The audio encoder processes mel spectrograms through a Conv2d stem
    (8x temporal downsampling) followed by transformer encoder layers.

    Defaults correspond to the 1.7B model.
    """

    num_mel_bins: int = 128
    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    d_model: int = 1024
    output_dim: int = 2048
    max_source_positions: int = 1500
    n_window: int = 50
    n_window_infer: int = 800
    conv_chunksize: int = 500
    downsample_hidden_size: int = 480
    activation_function: str = "gelu"
    dropout: float = 0.0

    @classmethod
    def for_0_6b(cls) -> AudioEncoderConfig:
        """Return config for the 0.6B model's audio encoder."""
        return cls(
            num_mel_bins=128,
            encoder_layers=18,
            encoder_attention_heads=14,
            encoder_ffn_dim=3584,
            d_model=896,
            output_dim=1024,
            max_source_positions=1500,
            n_window=50,
            n_window_infer=800,
            conv_chunksize=500,
            downsample_hidden_size=480,
            activation_function="gelu",
            dropout=0.0,
        )

    @classmethod
    def from_dict(cls, d: dict) -> AudioEncoderConfig:
        """Create config from a dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class TextDecoderConfig:
    """Configuration for the Qwen3-ASR text decoder.

    The text decoder is a Qwen3-style transformer with Multi-dimensional
    RoPE (MRoPE) that autoregressively generates transcription tokens.
    Uses RMSNorm (no bias), unlike the audio encoder which uses LayerNorm.

    Defaults correspond to the 1.7B model (GQA with 16 heads, 8 KV heads).
    """

    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 65536
    rope_theta: float = 1000000.0
    rope_scaling: Optional[dict] = None
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    tie_word_embeddings: bool = True

    @classmethod
    def for_0_6b(cls) -> TextDecoderConfig:
        """Return config for the 0.6B model's text decoder (uses GQA)."""
        return cls(
            vocab_size=151936,
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            hidden_act="silu",
            max_position_embeddings=65536,
            rope_theta=1000000.0,
            rope_scaling=None,
            rms_norm_eps=1e-6,
            attention_bias=False,
            tie_word_embeddings=True,
        )

    @classmethod
    def from_dict(cls, d: dict) -> TextDecoderConfig:
        """Create config from a dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class Qwen3ASRConfig:
    """Top-level configuration for the Qwen3-ASR model.

    Combines audio encoder and text decoder configs with model-level
    settings like special token IDs and supported languages.

    Args:
        audio_config: Configuration for the audio encoder.
        text_config: Configuration for the text decoder.
        audio_token_id: Token ID for the audio placeholder in the input
            sequence, replaced by audio features during forward pass.
        audio_start_token_id: Token ID marking the start of audio content.
        user_token_id: Token ID for the user role marker.
        support_languages: Optional list of supported language codes.
            If None, all languages are supported.
    """

    audio_config: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    text_config: TextDecoderConfig = field(default_factory=TextDecoderConfig)
    audio_token_id: int = 151676        # <|audio_pad|>
    audio_start_token_id: int = 151669  # <|audio_start|>
    audio_end_token_id: int = 151670    # <|audio_end|>
    user_token_id: int = 872
    support_languages: Optional[list] = None
    classify_num: Optional[int] = None
    timestamp_token_id: Optional[int] = None
    timestamp_segment_time: Optional[float] = None

    @classmethod
    def from_dict(cls, d: dict) -> Qwen3ASRConfig:
        """Create config from a HuggingFace config.json dictionary.

        The HF config format nests audio and text configs under a
        "thinker_config" key:

            {
                "thinker_config": {
                    "audio_config": { ... },
                    "text_config": { ... },
                    "audio_token_id": 151676,
                    "audio_start_token_id": 151669,
                    ...
                },
                "support_languages": [...],
            }

        This method handles both the nested HF format and a flat format
        where audio_config/text_config are at the top level.

        Args:
            d: Dictionary from config.json (or equivalent).

        Returns:
            Fully instantiated Qwen3ASRConfig.
        """
        # HF format: configs are nested under "thinker_config"
        # Token IDs live inside thinker_config, not at top level
        if "thinker_config" in d:
            thinker = d["thinker_config"]
            audio_dict = thinker.get("audio_config", {})
            text_dict = thinker.get("text_config", {})
            token_source = thinker
        else:
            audio_dict = d.get("audio_config", {})
            text_dict = d.get("text_config", {})
            token_source = d

        # Parse nested configs — if they're already dataclasses, use them directly
        if isinstance(audio_dict, dict):
            audio_config = AudioEncoderConfig.from_dict(audio_dict)
        else:
            audio_config = audio_dict

        if isinstance(text_dict, dict):
            text_config = TextDecoderConfig.from_dict(text_dict)
        else:
            text_config = text_dict

        return cls(
            audio_config=audio_config,
            text_config=text_config,
            audio_token_id=token_source.get("audio_token_id", 151676),
            audio_start_token_id=token_source.get("audio_start_token_id", 151669),
            audio_end_token_id=token_source.get("audio_end_token_id", 151670),
            user_token_id=token_source.get("user_token_id", 872),
            support_languages=d.get("support_languages"),
            classify_num=token_source.get("classify_num", d.get("classify_num")),
            timestamp_token_id=d.get("timestamp_token_id"),
            timestamp_segment_time=d.get("timestamp_segment_time"),
        )

    @classmethod
    def for_0_6b(cls) -> Qwen3ASRConfig:
        """Return config for the 0.6B model."""
        return cls(
            audio_config=AudioEncoderConfig.for_0_6b(),
            text_config=TextDecoderConfig.for_0_6b(),
        )

    @classmethod
    def for_1_7b(cls) -> Qwen3ASRConfig:
        """Return config for the 1.7B model (default)."""
        return cls(
            audio_config=AudioEncoderConfig(),
            text_config=TextDecoderConfig(),
        )
