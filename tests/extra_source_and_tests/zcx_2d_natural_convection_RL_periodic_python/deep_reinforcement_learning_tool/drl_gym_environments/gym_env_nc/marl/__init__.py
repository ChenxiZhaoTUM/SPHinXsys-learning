from .NCMAEnv import NCMAEnvParallel


def parallel_env(**kwargs):
    """PettingZoo 并行环境工厂：返回一个 ParallelEnv 实例。"""
    return NCMAEnvParallel(**kwargs)


def raw_env(**kwargs):
    """可选：把并行环境包装成 AEC 接口（部分 MARL 库更喜欢这个）。"""
    from pettingzoo.utils import parallel_to_aec
    return parallel_to_aec(parallel_env(**kwargs))


__all__ = ["parallel_env", "raw_env", "NCMAEnvParallel"]
