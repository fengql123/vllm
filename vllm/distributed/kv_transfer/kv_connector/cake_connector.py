from typing import TYPE_CHECKING, Union

import torch
from torch.distributed import ProcessGroup

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.lmcache_connector import LMCacheConnector
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.distributed import get_tp_group

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class CakeConnector(KVConnectorBase):
    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):
        
        from lmcache.integration.vllm.cake_adapter import cake_pre_process, cake_preprocess_merge, close_cake_adapter, init_cake_adapter
        self.transfer_config = config.kv_transfer_config
        self.vllm_config = config
        self.lmcache_connector = LMCacheConnector(
            rank=rank,
            local_rank=local_rank,
            config=config,
        )
        init_cake_adapter(self.lmcache_connector.engine, get_tp_group().cpu_group, local_rank)
        self.hidden_state_cache_engine = HiddenStateCacheEngine.build_from_lmc_engine(self.lmcache_connector.engine)
        self.cake_pre_process = cake_pre_process
        self.cake_preprocess_merge = cake_preprocess_merge
        self.close_cake_adapter = close_cake_adapter
        self.model_config = config.model_config
        self.parallel_config = config.parallel_config
        self.cache_config = config.cache_config
        
    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor],
        cpu_group: ProcessGroup
    ) -> tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        retrieve_status = self.lmcache_connector.lmcache_should_retrieve(model_input, cpu_group)
        self.cake_pre_process(model_input, kv_caches, retrieve_status)
        return None, False, model_input
    
    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
        cpu_group: ProcessGroup
    ) -> None:
        store_status = self.lmcache_connector.lmcache_should_store(model_input, cpu_group)
        self.lmcache_connector.lmcache_store_kv(
            self.model_config,
            self.parallel_config,
            self.cache_config,
            model_executable,
            model_input,
            kv_caches,
            store_status,
        )

    def close(self):
        self.lmcache_connector.lmcache_engine_builder.destroy(self.lmcache_connector.lmcache_engine_name)
        self.close_cake_adapter()
        
    def retrieve_seqgroup_hidden_states(self,model_input):
        return self.hidden_state_cache_engine.retrieve(model_input)

    def store_seqgroup_hidden_states(self,hidden_states,model_input):
        self.hidden_state_cache_engine.store(hidden_states,model_input)
        

from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.integration.vllm.utils import ENGINE_NAME

class HiddenStateCacheEngine:
    _FMT: str = "hidden_state"          # new “format” tag; re-uses LMCache key logic

    def __init__(self, engine: LMCacheEngine | None = None) -> None:
        # Re-use the KV-cache engine if it already exists; otherwise throw --
        # users should have initialised it once via `init_lmcache_engine`.
        
        if engine is None:
            engine = LMCacheEngineBuilder.get(ENGINE_NAME)
            if engine is None:
                raise RuntimeError(
                    "LMCacheEngine not initialised — call init_lmcache_engine(...) "
                    "before constructing HiddenStateCacher"
                )
        self.engine: LMCacheEngine = engine
        if self.engine.chunk_size != 0:
            logger.debug(
                "Hidden-state cacher created (chunk_size=%d).", self.engine.chunk_size
            )
    @classmethod
    def build_from_lmc_engine(cls,engine: LMCacheEngine):
        return cls(engine)

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #

    @torch.inference_mode()
    def store(
        self,
        hidden_states: torch.Tensor,
        model_input: "ModelInputForGPUWithSamplingMetadata"
    ) -> None:
        """Segment *hidden_states* by sequence and persist each slice.

        Args
        ----
        model_input:
            The exact input object passed to vLLM’s model runner.
        hidden_states:
            Shape **[num_tokens, hidden_dim]**, where *num_tokens* ==
            `len(model_input.input_tokens)`.
        """
        seq_tokens, seq_slices = self._segment_hidden_states(model_input, hidden_states)

        # Build (key, tensor) pairs for batched put.
        put_pairs: List[Tuple[CacheEngineKey, torch.Tensor]] = []
        for tokens, h_slice in zip(seq_tokens, seq_slices, strict=False):
            key = self._make_key(tokens)
            put_pairs.append((key, h_slice.cpu().contiguous()))

        n = self.engine.engine_.batched_put(put_pairs, blocking=True)
        logger.debug("Hidden-state cacher: stored %d sequence slices.", n)

    @torch.inference_mode()
    def retrieve(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        device: torch.device | None = None,
    ) -> Tuple[torch.Tensor | None, torch.BoolTensor]:
        """Attempt to fetch cached hidden states for every sequence.

        Returns
        -------
        merged_hs  : `torch.Tensor | None`
            *None* if **any** sequence slice is missing (caller must run a forward).
            Otherwise the concatenated tensor exactly matching the
            `[num_tokens, hidden_dim]` layout expected by the caller.
        hit_mask   : `torch.BoolTensor`
            Per-token bool mask (True ==> token’s hidden state is cached & returned).
        """
        seq_tokens, _ = self._collect_sequence_tokens(model_input)
        cached_slices: List[torch.Tensor] = []
        hits: List[bool] = []

        for tokens in seq_tokens:
            obj = self.engine.engine_.get(self._make_key(tokens))
            if obj is None:
                hits.extend([False] * len(tokens))
                cached_slices.append(None)          # type: ignore[arg-type]
            else:
                hits.extend([True] * len(tokens))
                cached_slices.append(obj)

        hit_tensor = torch.tensor(hits, dtype=torch.bool, device="cpu")

        # If *any* slice is missing we short-circuit (common decision pattern
        # in cache-before-forward hooks).
        if any(s is None for s in cached_slices):
            return None, hit_tensor

        merged = torch.cat(cached_slices, dim=0).to(device or cached_slices[0].device)
        return merged, hit_tensor

    # --------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # --------------------------------------------------------------------- #

    def _make_key(self, tokens: torch.Tensor) -> CacheEngineKey:
        """Use LMCacheEngine’s SHA-256 prefix hash for deterministic keys."""
        # The engine’s helper returns *intermediate* hashes chunk-by-chunk; for
        # a single sequence we just need one SHA-256 over the entire token list.
        token_hash = self.engine._hash(tokens.cpu(), self.engine._get_init_hash())
        return self.engine._make_key(token_hash, self._FMT)

    @staticmethod
    def _collect_sequence_tokens(
        model_input: "ModelInputForGPUWithSamplingMetadata",
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """Return (`tokens_per_seq`, `lengths_per_seq`)."""
        assert model_input.sampling_metadata is not None
        seq_groups = model_input.sampling_metadata.seq_groups
        assert seq_groups is not None

        tokens_flat = model_input.input_tokens       # [num_tokens] *CPU/CUDA*

        tokens_per_seq: List[torch.Tensor] = []
        lengths: List[int] = []

        cursor = 0
        for group in seq_groups:
            # Dict preserves insertion order (py3.7+) — matches vLLM token order.
            for seq_id, seq_data in group.seq_data.items():
                n = seq_data.get_len()
                tokens_per_seq.append(tokens_flat[cursor : cursor + n].cpu())
                lengths.append(n)
                cursor += n

        # Sanity: consumed all tokens?
        assert cursor == tokens_flat.shape[0], (
            f"Segment logic consumed {cursor} tokens, expected "
            f"{tokens_flat.shape[0]}"
        )
        return tokens_per_seq, lengths

    def _segment_hidden_states(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        hidden_states: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return (tokens_per_seq, hidden_state_slices_per_seq)."""
        tokens_per_seq, lengths = self._collect_sequence_tokens(model_input)

        slices: List[torch.Tensor] = []
        offset = 0
        for n in lengths:
            slices.append(hidden_states[offset : offset + n].cpu())
            offset += n
        assert offset == hidden_states.shape[0]
        return tokens_per_seq, slices
