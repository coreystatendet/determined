from datetime import timedelta
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import weakref

import ray
from ray.air._internal.util import find_free_port
from ray.util import get_node_ip_address
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import torch

REMOTE_PARAM_REGISTRY = {}


def remote_method(
    distribute: Optional[Callable] = None, amalgamate: Optional[Callable] = None
) -> Callable:
    """
    Specifies how to distribute arguments and amalgamate results of distributed method calls.

    For this demo, these functions are applied to each argument independently.  A more complex API
    would be needed to support multiple arguments with different behavior, keyword vs non-keyword
    arguments, etc.

    distribute should take in number of slots and a single value return a list of that length.
    amalgamate s
    """

    def _remote_method_decorator(func: Callable) -> Callable:
        REMOTE_PARAM_REGISTRY[func.__qualname__] = {
            "distribute": distribute,
            "amalgamate": amalgamate,
        }

        return func

    return _remote_method_decorator


class RemoteModelProxy:
    def __init__(
        self, actor_class: type, num_slots: int, node_size: int, *args: List, **kwargs: Dict
    ) -> None:
        self.actor_class = actor_class
        self.num_slots = num_slots
        self.node_size = node_size
        self.args = args
        self.kwargs = kwargs
        self.actors: List[Any] = []
        self._launch_actors()

    def _launch_actors(self) -> None:
        assert (self.num_slots % self.node_size) == 0 or (self.num_slots < self.node_size)

        @ray.remote(num_gpus=1, num_cpus=0)
        class Wrapped(PyTorchDistributedWorkerMixin, self.actor_class):  # type: ignore
            pass

        initialize_futures = []
        pgs = []
        num_nodes = max(1, self.num_slots // self.node_size)
        local_world_size = min(self.num_slots, self.node_size)
        for i in range(num_nodes):
            print(f"Allocating placement group {i}")
            pg = placement_group([{"GPU": local_world_size}], strategy="STRICT_PACK")
            weakref.finalize(self, remove_placement_group, pg)
            # TODO: Parallelize?
            pgs.append(pg)
            ray.get(pg.ready())
        for i in range(num_nodes):
            pg = pgs[i]
            for j in range(local_world_size):
                rank = i * self.node_size + j
                # Need to finish PyTorch Distributed initialize before calling
                #   class __init__, since it may depend on PyTorch Distributed!
                actor = Wrapped.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)
                ).remote()
                self.actors.append(actor)
                if rank == 0:
                    master_addr, master_port = ray.get(actor._get_c10d_connection_info.remote())
                initialize_futures.append(
                    actor._initialize.remote(
                        rank=rank,
                        local_rank=j,
                        world_size=self.num_slots,
                        local_world_size=local_world_size,
                        cross_rank=i,
                        cross_size=num_nodes,
                        master_addr=master_addr,
                        master_port=master_port,
                        remote_param_registry=REMOTE_PARAM_REGISTRY,
                    )
                )
        ray.wait(initialize_futures)

    def __getattr__(self, name: str) -> Callable:
        if not (hasattr(self.actors[0], name)):
            raise Exception(f"{name} undefined on {self.actor_class.__name__}")

        def call_on_actors(*args: List, **kwargs: Dict) -> CompoundFuture:
            qualname = self.actor_class.__name__ + "." + name
            if qualname in REMOTE_PARAM_REGISTRY:
                distribute = REMOTE_PARAM_REGISTRY[qualname]["distribute"]
                amalgamate = REMOTE_PARAM_REGISTRY[qualname]["amalgamate"]
            else:
                distribute = None
                amalgamate = None
            futures = []
            if distribute:
                distributed_args = [distribute(self.num_slots, arg) for arg in args]
                distributed_kwargs = {k: distribute(self.num_slots, v) for k, v in kwargs.items()}
            for rank in range(len(self.actors)):
                if distribute:
                    rank_args = [arg[rank] for arg in distributed_args]
                    rank_kwargs = {k: v[rank] for k, v in distributed_kwargs.items()}
                else:
                    rank_args = args
                    rank_kwargs = kwargs
                futures.append(getattr(self.actors[rank], name).remote(*rank_args, **rank_kwargs))
            return CompoundFuture(futures, amalgamate)

        return call_on_actors


class CompoundFuture:
    def __init__(self, ray_futures: List[int], amalgamate_fn: Optional[Callable]) -> None:
        self.ray_futures = ray_futures
        self.amalgamate_fn = amalgamate_fn

    def wait(self) -> None:
        ray.wait(self.ray_futures)

    def get(self) -> Any:
        res = ray.get(self.ray_futures)
        if self.amalgamate_fn:
            res = self.amalgamate_fn(res)
        return res


# TODO: Eventually add launcher and run_type arguments.
# For now, everything is a PyTorch Distributed experiment.
def remote_model(
    actor_class: type, *args: List, num_slots: int = 1, node_size: int = 1, **kwargs: Dict
) -> RemoteModelProxy:
    return RemoteModelProxy(actor_class, num_slots, node_size, *args, **kwargs)


class PyTorchDistributedWorkerMixin:
    def _get_c10d_connection_info(self) -> Tuple[str, int]:
        return (get_node_ip_address(), find_free_port())

    def _initialize(
        self,
        rank: int,
        local_rank: int,
        world_size: int,
        local_world_size: int,
        cross_rank: int,
        cross_size: int,
        master_addr: str,
        master_port: int,
        remote_param_registry: Dict,
    ) -> None:
        # Decorators don't re-execute on actors, so keep the param registry in sync to support
        # nested remote models.
        global REMOTE_PARAM_REGISTRY
        REMOTE_PARAM_REGISTRY.update(remote_param_registry)
        # os.environ["NCCL_DEBUG"] = "INFO"
        # os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
        # NOTE: Getting weird errors around P2P -- would need to resolve this in production.
        # Workaround for now by disabling
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)
        os.environ["GROUP_RANK"] = str(cross_rank)
        os.environ["GROUP_WORLD_SIZE"] = str(cross_size)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=1800),
        )  # type: ignore
        self._rank = rank
        self._world_size = world_size


def split_batch(num_slots: int, x: torch.Tensor):
    batch_len = x.shape[0] // num_slots
    xs = []
    for i in range(num_slots):
        start = i * batch_len
        end = (i + 1) * batch_len
        xs.append(x[start:end, ...])
    return xs


def split_batch_list(num_slots: int, x: List):
    batch_len = len(x) // num_slots
    xs = []
    for i in range(num_slots):
        start = i * batch_len
        end = (i + 1) * batch_len
        xs.append(x[start:end])
    return xs
