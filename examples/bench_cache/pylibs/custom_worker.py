"""A Custom GPU worker class."""
from typing import Dict, List, Optional, Set, Tuple, Type
import time
import torch
import torch.distributed

import vllm.envs as envs
from vllm.attention import get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, VllmConfig)
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.sequence import ExecuteModelRequest
from vllm.worker.model_runner import GPUModelRunnerBase, ModelRunner
from vllm.sequence import (ExecuteModelRequest, IntermediateTensors,
                           SequenceGroupMetadata, SequenceGroupMetadataDelta)
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, bind_kv_cache
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase, WorkerBase,
                                     WorkerInput)
from vllm.worker.worker import Worker
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.config import ObservabilityConfig, VllmConfig
from vllm.distributed import broadcast_tensor_dict, get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, IntermediateTensors
from vllm.utils import (enable_trace_function_call_for_thread,
                        resolve_obj_by_qualname, update_environment_variables)

logger = init_logger(__name__)


class CustomGPUWorker(Worker):
    """ A worker class that executes (a partition of) the model on a GPU.
    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[GPUModelRunnerBase]] = None,
    ) -> None:
        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker,
                         model_runner_cls=model_runner_cls)

    
    
    def _execute_model_spmd(
        self,
        execute_model_req: ExecuteModelRequest,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Optional[List[SamplerOutput]]:

        
        if execute_model_req is not None:
            new_seq_group_metadata_list = self._get_cached_seq_group_metadata(
                execute_model_req.seq_group_metadata_list,
                execute_model_req.finished_requests_ids)

            execute_model_req.seq_group_metadata_list = (
                new_seq_group_metadata_list)
        output = super()._execute_model_spmd(execute_model_req,
                                             intermediate_tensors)
        return output
    
    
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[List[SamplerOutput]]:
        """Executes at least one model step on the given sequences, unless no
        sequences are provided.
        
        Here we add a custom kv cache to the model.
        """
        # # prompt的token
        # print(execute_model_req.seq_group_metadata_list[0].seq_data[0].prompt_token_ids)
        # # 生成的token
        # print(execute_model_req.seq_group_metadata_list[0].seq_data[0].output_token_ids)
        
        start_time = time.perf_counter()

        inputs = self.prepare_input(execute_model_req)
        if inputs is None:
            return None

        model_input, worker_input, kwargs = inputs
        num_steps = worker_input.num_steps

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        intermediate_tensors = None
        orig_model_execute_time = 0.0
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group()))
            if (self.observability_config is not None
                    and self.observability_config.collect_model_execute_time):
                orig_model_execute_time = intermediate_tensors.tensors.get(
                    "model_execute_time", torch.tensor(0)).item()

        output = self.model_runner.execute_model(
            model_input=model_input,
            kv_caches=self.kv_cache[worker_input.virtual_engine]
            if self.kv_cache is not None else None,
            intermediate_tensors=intermediate_tensors,
            num_steps=num_steps,
            **kwargs,
        )

        model_execute_time = time.perf_counter() - start_time
        if not get_pp_group().is_last_rank:
            # output is IntermediateTensors
            assert isinstance(output, IntermediateTensors)
            if (self.observability_config is not None
                    and self.observability_config.collect_model_execute_time):
                output.tensors["model_execute_time"] = torch.tensor(
                    model_execute_time + orig_model_execute_time)
            get_pp_group().send_tensor_dict(output.tensors,
                                            all_gather_group=get_tp_group())
            return [None]
        if (self.observability_config is not None
                and self.observability_config.collect_model_execute_time
                and output is not None):
            for o in output:
                o.model_execute_time = (orig_model_execute_time +
                                        model_execute_time)

        # output is List[SamplerOutput]
        return output