"""Task registry system for managing environment registration and creation."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class _TaskCfg:
  env_cfg: ManagerBasedRlEnvCfg | Callable[[], ManagerBasedRlEnvCfg]
  play_env_cfg: ManagerBasedRlEnvCfg
  rl_cfg: RslRlBaseRunnerCfg
  runner_cls: type | None


# Private module-level registry: task_id -> task config.
_REGISTRY: dict[str, _TaskCfg] = {}


def register_mjlab_task(
  task_id: str,
  env_cfg: ManagerBasedRlEnvCfg | Callable[[], ManagerBasedRlEnvCfg],
  play_env_cfg: ManagerBasedRlEnvCfg,
  rl_cfg: RslRlBaseRunnerCfg,
  runner_cls: type | None = None,
) -> None:
  """Register an environment task.

  Args:
    task_id: Unique task identifier (e.g., "Mjlab-Velocity-Rough-Unitree-Go1").
    env_cfg: Training configuration or a factory returning a fresh configuration.
    play_env_cfg: Environment configuration in "play" mode.
    rl_cfg: RL runner configuration.
    runner_cls: Optional custom runner class. If None, uses OnPolicyRunner.
  """
  if task_id in _REGISTRY:
    raise ValueError(f"Task '{task_id}' is already registered")
  _REGISTRY[task_id] = _TaskCfg(env_cfg, play_env_cfg, rl_cfg, runner_cls)


def list_tasks() -> list[str]:
  """List all registered task IDs."""
  return sorted(_REGISTRY.keys())


def load_env_cfg(task_name: str, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Load environment configuration for a task.

  Returns a fresh configuration from the factory or a copy of the template.
  """
  task = _REGISTRY[task_name]
  config = task.play_env_cfg if play else task.env_cfg
  return config() if callable(config) else deepcopy(config)


def load_rl_cfg(task_name: str) -> RslRlBaseRunnerCfg:
  """Load RL configuration for a task.

  Returns a fresh configuration from the factory or a copy of the template.
  """
  return deepcopy(_REGISTRY[task_name].rl_cfg)


def load_runner_cls(task_name: str) -> type | None:
  """Load the runner class for a task.

  If None, the default OnPolicyRunner will be used.
  """
  return _REGISTRY[task_name].runner_cls
