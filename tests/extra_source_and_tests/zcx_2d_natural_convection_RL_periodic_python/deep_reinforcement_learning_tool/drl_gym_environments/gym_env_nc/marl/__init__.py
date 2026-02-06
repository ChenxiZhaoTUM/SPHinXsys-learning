"""
gym_env_nc/marl/__init__.py

This package provides a Gymnasium Env implementation for a Vignon-style
translation-invariant "multi-agent" setup (implemented as vectorized pseudo-envs).

Important
---------
- `NCPseudoEnv` is a standard Gymnasium Env.
- This module does NOT provide a PettingZoo ParallelEnv/AEC interface; we keep
  factory names (`parallel_env`, `raw_env`) only for compatibility with upstream
  import chains that expect them.
"""

from __future__ import annotations
from .NCPseudoEnv import NCPseudoEnv


def parallel_env(**kwargs):
    """Compatibility factory returning a Gymnasium Env (NCPseudoEnv)."""
    return NCPseudoEnv(**kwargs)


def raw_env(**kwargs):
    """Compatibility alias."""
    return parallel_env(**kwargs)


__all__ = ["parallel_env", "raw_env", "NCPseudoEnv"]
