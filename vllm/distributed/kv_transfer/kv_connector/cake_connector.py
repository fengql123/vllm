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
        