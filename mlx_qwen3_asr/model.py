"""Top-level Qwen3-ASR model composition and compatibility exports.

Qwen3-ASR is an encoder-decoder model for automatic speech recognition:
  - Audio Encoder: Conv2d stem (8x downsample) -> sinusoidal pos -> transformer
  - Text Decoder: Embedding -> transformer with MRoPE -> LM head
  - Top-level: Audio features injected into text embeddings at placeholder positions

Weight naming follows the HuggingFace checkpoint layout so that
``mx.load`` / ``model.load_weights`` works without key remapping.
"""

from __future__ import annotations

from typing import Optional, cast

import mlx.core as mx
import mlx.nn as nn

from .attention import _scaled_dot_product_attention, scalar_int
from .config import Qwen3ASRConfig
from .decoder import (
    KVCache,
    SwiGLU,
    TextAttention,
    TextDecoder,
    TextDecoderLayer,
    _create_causal_mask,
)
from .encoder import (
    AudioAttention,
    AudioEncoder,
    AudioEncoderLayer,
    SinusoidalPositionEmbedding,
    _apply_windowed_encoder_layers,
    _create_windowed_mask,
)


class Qwen3ASRModel(nn.Module):
    """Top-level Qwen3-ASR model: audio encoder + text decoder + LM head.

    During inference:
      1. Encode audio via the audio tower.
      2. Build text embeddings, injecting audio features at placeholder positions.
      3. Run the text decoder autoregressively.
      4. Project to vocabulary logits with the LM head.

    Weight key mapping (matches HuggingFace checkpoint):
      - ``audio_tower.*``  -- AudioEncoder
      - ``model.*``        -- TextDecoder (embed_tokens, layers, norm)
      - ``lm_head.*``      -- final linear projection

    Args:
        config: Qwen3ASRConfig with full model hyperparameters.
    """

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__()
        self.config = config
        self.audio_token_id = config.audio_token_id
        output_size = config.classify_num or config.text_config.vocab_size

        self.audio_tower = AudioEncoder(config.audio_config)
        self.model = TextDecoder(config.text_config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            output_size,
            bias=False,
        )

        # Tie weights if configured
        if (
            config.text_config.tie_word_embeddings
            and output_size == config.text_config.vocab_size
        ):
            self.lm_head.weight = self.model.embed_tokens.weight

    def __call__(
        self,
        input_ids: mx.array,
        input_features: Optional[mx.array] = None,
        feature_lens: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        """Forward pass through the full model.

        Args:
            input_ids: Token IDs including audio placeholders, shape (B, L).
            input_features: Mel spectrogram, shape (B, n_mels, n_frames).
                Required on the first call (prefill); None during generation.
            feature_lens: Audio frame counts, shape (B,).
                Required when input_features is provided.
            attention_mask: Optional causal mask.
            position_ids: MRoPE position IDs, shape (B, 3, L).
            cache: Optional KV cache for autoregressive generation.

        Returns:
            Logits, shape (B, L, vocab_size).
        """
        # Get text embeddings
        embeds = self._embed_tokens(input_ids, validate_input_ids=True)

        # Encode and inject audio features if audio is provided
        if input_features is not None:
            if feature_lens is None:
                raise ValueError("feature_lens is required when input_features is provided")
            audio_features, _ = self.audio_tower(input_features, feature_lens)
            audio_mask = cast(mx.array, input_ids == self.audio_token_id)
            embeds = self._inject_audio_features(embeds, audio_features, audio_mask)

        # Run text decoder
        hidden = self.model(
            inputs_embeds=embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache=cache,
        )

        # Project to vocabulary logits
        logits = self.lm_head(hidden)
        return logits

    def _validate_input_ids_for_embed(self, input_ids: mx.array) -> None:
        """Validate token IDs before embedding lookup."""
        self._validate_input_ids_dtype(input_ids)
        if input_ids.size == 0:
            return

        self._validate_input_ids_range(input_ids)

    def _validate_input_ids_dtype(self, input_ids: mx.array) -> None:
        """Validate token dtype for embedding lookup."""
        if not mx.issubdtype(input_ids.dtype, mx.integer):
            raise ValueError(
                "input_ids must use an integer dtype for embed_tokens, "
                f"got dtype={input_ids.dtype}"
            )

    def _validate_input_ids_range(self, input_ids: mx.array) -> None:
        """Validate token ID bounds for embedding lookup."""
        vocab_size = int(self.config.text_config.vocab_size)
        min_id = scalar_int(mx.min(input_ids))
        max_id = scalar_int(mx.max(input_ids))
        if min_id < 0 or max_id >= vocab_size:
            raise ValueError(
                "input_ids out of bounds for embed_tokens: "
                f"min_token_id={min_id}, max_token_id={max_id}, vocab_size={vocab_size}"
            )

    def _embed_tokens(
        self,
        input_ids: mx.array,
        *,
        validate_input_ids: bool = True,
    ) -> mx.array:
        """Embed token IDs with optional strict-range validation."""
        if validate_input_ids:
            self._validate_input_ids_for_embed(input_ids)
        else:
            # Decode fast path: retain dtype safety and skip range scans that
            # force host synchronization on every token step.
            self._validate_input_ids_dtype(input_ids)
        return self.model.embed_tokens(input_ids)

    def _inject_audio_features(
        self,
        embeds: mx.array,
        audio_features: mx.array,
        audio_mask: mx.array,
    ) -> mx.array:
        """Replace audio placeholder positions with encoded audio features.

        Uses cumulative-sum indexing to map each placeholder position to the
        corresponding audio feature vector, then selects via ``mx.where``.

        Args:
            embeds: Text embeddings, shape (B, L, D). Positions matching
                audio_token_id contain placeholder embeddings to replace.
            audio_features: Encoded audio, shape (B, N_audio, D) where
                N_audio is the number of audio tokens produced by the encoder.
            audio_mask: Boolean mask, shape (B, L). True where
                input_ids == audio_token_id.

        Returns:
            Embeddings with audio features injected, shape (B, L, D).
        """
        if (
            embeds.shape[0] != audio_features.shape[0]
            or embeds.shape[0] != audio_mask.shape[0]
            or embeds.shape[1] != audio_mask.shape[1]
        ):
            raise ValueError(
                "Audio injection shape mismatch: "
                f"embeds={embeds.shape}, "
                f"audio_features={audio_features.shape}, "
                f"audio_mask={audio_mask.shape}"
            )
        if embeds.shape[2] != audio_features.shape[2]:
            raise ValueError(
                "Audio injection channel mismatch: "
                f"embed_dim={embeds.shape[2]}, "
                f"audio_feature_dim={audio_features.shape[2]}"
            )

        audio_counts = mx.sum(audio_mask.astype(mx.int32), axis=1)
        max_count = scalar_int(mx.max(audio_counts))
        max_audio_tokens = int(audio_features.shape[1])
        if max_count == 0:
            # Nothing to inject; avoid gather() on empty audio feature tensors.
            return embeds
        if max_count > max_audio_tokens:
            raise ValueError(
                "Audio injection out of bounds: "
                f"max_audio_placeholders={max_count}, "
                f"audio_features_tokens={max_audio_tokens}"
            )

        # Build a cumulative index that maps each position to an audio feature index.
        # For positions where audio_mask is True, cum_idx increases by 1.
        # For non-audio positions, cum_idx stays the same (pointing to the last
        # valid audio feature, but those positions won't be selected anyway).
        cum_idx = mx.cumsum(audio_mask.astype(mx.int32), axis=1) - 1
        cum_idx = mx.maximum(cum_idx, 0)  # (B, L)
        masked_cum_idx = mx.where(audio_mask, cum_idx, mx.zeros_like(cum_idx))
        max_masked_idx = scalar_int(mx.max(masked_cum_idx))
        if max_masked_idx >= max_audio_tokens:
            raise AssertionError(
                "Audio injection index overflow: "
                f"max_cum_idx={max_masked_idx}, "
                f"audio_features_tokens={max_audio_tokens}. "
                "This indicates prompt/audio token count drift."
            )

        # Gather audio features for every position using cum_idx.
        # audio_features[b, cum_idx[b]] gives shape (L, D) for each batch element.
        B = embeds.shape[0]
        audio_expanded_parts = []
        for b in range(B):
            expanded = audio_features[b, cum_idx[b]]  # (L, D)
            audio_expanded_parts.append(expanded)
        audio_expanded = mx.stack(audio_expanded_parts)  # (B, L, D)

        # Select audio features at masked positions, text embeds elsewhere
        mask_3d = audio_mask[:, :, None]  # (B, L, 1)
        result = mx.where(mask_3d, audio_expanded, embeds)
        return result

    def prefill(
        self,
        input_ids: mx.array,
        audio_features: mx.array,
        position_ids: mx.array,
        cache: KVCache,
    ) -> mx.array:
        """Prefill decode cache from prompt + injected audio features.

        Args:
            input_ids: Prompt token IDs including audio placeholders, shape (B, L).
            audio_features: Encoded audio features, shape (B, N_audio, D).
            position_ids: MRoPE position IDs for the prompt, shape (B, 3, L).
            cache: KV cache to populate in-place.

        Returns:
            Logits for the final prompt position, shape (B, 1, vocab_size).
        """
        embeds = self._embed_tokens(input_ids, validate_input_ids=True)
        audio_mask = cast(mx.array, input_ids == self.audio_token_id)
        embeds = self._inject_audio_features(embeds, audio_features, audio_mask)

        hidden = self.model(
            inputs_embeds=embeds,
            position_ids=position_ids,
            attention_mask=None,
            cache=cache,
        )
        return self.lm_head(hidden[:, -1:, :])

    def step(
        self,
        input_ids: mx.array,
        position_ids: mx.array,
        cache: KVCache,
        *,
        validate_input_ids: bool = True,
    ) -> mx.array:
        """Decode one autoregressive step with an existing cache.

        Args:
            input_ids: Next-token IDs, shape (B, 1).
            position_ids: Step MRoPE position IDs, shape (B, 3, 1).
            cache: KV cache to update in-place.
            validate_input_ids: Whether to run strict token-range validation.
                External callers should keep this enabled. Generation hot paths
                may disable it to reduce per-token overhead.

        Returns:
            Step logits, shape (B, 1, vocab_size).
        """
        embeds = self._embed_tokens(
            input_ids,
            validate_input_ids=validate_input_ids,
        )
        hidden = self.model(
            inputs_embeds=embeds,
            position_ids=position_ids,
            attention_mask=None,
            cache=cache,
        )
        return self.lm_head(hidden)

    def step_many(
        self,
        input_ids: mx.array,
        position_ids: mx.array,
        cache: KVCache,
        *,
        validate_input_ids: bool = True,
    ) -> mx.array:
        """Decode multiple autoregressive steps with an existing cache.

        Args:
            input_ids: Token IDs to process, shape (B, T), where T >= 1.
            position_ids: Step MRoPE position IDs, shape (B, 3, T).
            cache: KV cache to update in-place.
            validate_input_ids: Whether to run strict token-range validation.

        Returns:
            Step logits, shape (B, T, vocab_size).
        """
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids for step_many must be rank-2 (B, T), got shape={input_ids.shape}"
            )
        if position_ids.ndim != 3:
            raise ValueError(
                "position_ids for step_many must be rank-3 (B, 3, T), "
                f"got shape={position_ids.shape}"
            )
        if input_ids.shape[0] != position_ids.shape[0]:
            raise ValueError(
                "Batch mismatch in step_many: "
                f"input_ids={input_ids.shape}, position_ids={position_ids.shape}"
            )
        if input_ids.shape[1] != position_ids.shape[2]:
            raise ValueError(
                "Sequence mismatch in step_many: "
                f"input_ids={input_ids.shape}, position_ids={position_ids.shape}"
            )

        embeds = self._embed_tokens(
            input_ids,
            validate_input_ids=validate_input_ids,
        )
        hidden = self.model(
            inputs_embeds=embeds,
            position_ids=position_ids,
            attention_mask=None,
            cache=cache,
        )
        return self.lm_head(hidden)

    def create_cache(self, max_seq_len: Optional[int] = None) -> KVCache:
        """Create a fresh KV cache for autoregressive generation.

        Args:
            max_seq_len: Optional preallocation target for cache growth.

        Returns:
            Empty KVCache with the correct number of layers.
        """
        return KVCache(
            self.config.text_config.num_hidden_layers,
            max_seq_len=max_seq_len,
        )

    @property
    def num_audio_encoder_layers(self) -> int:
        """Return the number of audio encoder transformer layers."""
        return len(self.audio_tower.layers)

    @property
    def num_text_decoder_layers(self) -> int:
        """Return the number of text decoder transformer layers."""
        return len(self.model.layers)


__all__ = [
    "_scaled_dot_product_attention",
    "SinusoidalPositionEmbedding",
    "AudioAttention",
    "AudioEncoderLayer",
    "AudioEncoder",
    "_create_windowed_mask",
    "_apply_windowed_encoder_layers",
    "TextAttention",
    "SwiGLU",
    "TextDecoderLayer",
    "TextDecoder",
    "KVCache",
    "_create_causal_mask",
    "Qwen3ASRModel",
]
