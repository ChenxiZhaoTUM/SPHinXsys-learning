# gym_env_nc/marl/__init__.py

from .NCPseudoEnv import NCPseudoEnv


def parallel_env(**kwargs):
    """
    Factory required by gym_env_nc/__init__.py.
    NOTE: 这里返回的是 Gymnasium Env (NCPseudoEnv)，不是 PettingZoo ParallelEnv。
    你的 sac_multi.py 训练只用 NCPseudoEnv，因此这样即可满足导入链。
    """
    return NCPseudoEnv(**kwargs)


def raw_env(**kwargs):
    """
    保持接口名字存在，避免 gym_env_nc/__init__.py 导入失败。
    如果你未来真要 PettingZoo AEC，请另外实现一个 ParallelEnv 再在这里转换。
    """
    return parallel_env(**kwargs)


__all__ = ["parallel_env", "raw_env", "NCPseudoEnv"]
