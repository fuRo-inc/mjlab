from __future__ import annotations

import pathlib
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from rsl_rl.utils.logger import Logger


class MjlabLogger(Logger):
  """Logger with fuRo-specific curriculum status display."""

  def __init__(self, *args, env=None, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.env = env

  @staticmethod
  def _get_attr_any(env_obj, name: str, default=None):
    if env_obj is None:
      return default
    if hasattr(env_obj, name):
      return getattr(env_obj, name)
    for attr in ("_env", "env", "unwrapped", "venv"):
      if hasattr(env_obj, attr):
        base = getattr(env_obj, attr)
        if base is not None and hasattr(base, name):
          return getattr(base, name)
    return default

  @staticmethod
  def _show_extra_in_cli(key: str) -> bool:
    hidden_prefixes = (
      "Metrics/",
      "Debug/",
      "Curriculum/",
    )
    return not key.startswith(hidden_prefixes)

  def _log_curriculum_bar_image(
    self,
    scalar_values: dict[str, float],
    it: int,
    tag: str,
    title: str,
    xlabel: str,
    xlim: tuple[float, float] | None = None,
  ) -> None:
    if self.writer is None or not scalar_values:
      return

    names = list(scalar_values.keys())
    values = [scalar_values[name] for name in names]

    fig_height = max(3.0, 0.32 * len(names))
    fig, ax = plt.subplots(figsize=(9.0, fig_height))

    ax.barh(names, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    if xlim is not None:
      ax.set_xlim(*xlim)

    for i, value in enumerate(values):
      ax.text(value, i, f" {value:.2f}", va="center", fontsize=8)

    fig.tight_layout()
    self.writer.add_figure(tag, fig, it)  # type: ignore
    plt.close(fig)

  def _get_unlock_status(self) -> tuple[bool, float | None]:
    good_steps = self._get_attr_any(self.env, "lin_track_good_steps", None)
    unlock_steps = self._get_attr_any(self.env, "_uniform_reset_unlock_steps", 200)
    unlocked = bool(self._get_attr_any(self.env, "_uniform_reset_unlocked", False))
    progress = None
    if good_steps is not None:
      try:
        progress = float(good_steps.max().item()) / float(unlock_steps)
      except Exception:
        progress = None
    return unlocked, progress

  def _get_roll_pitch_scale(self) -> float | None:
    value = self._get_attr_any(self.env, "_roll_pitch_scale", None)
    if value is None:
      return None
    try:
      return float(value)
    except Exception:
      return None

  def _reward_color(self) -> tuple[str, str, str]:
    unlocked, progress = self._get_unlock_status()
    if unlocked:
      return "\033[92m", "\033[0m", "Unlocked"
    if progress is not None and progress >= 0.5:
      return "\033[93m", "\033[0m", f"Warming ({progress * 100:.0f}%)"
    if progress is not None:
      return "", "", f"Locked ({progress * 100:.0f}%)"
    return "", "", "Locked"

  def log(
    self,
    it: int,
    start_it: int,
    total_it: int,
    collect_time: float,
    learn_time: float,
    loss_dict: dict,
    learning_rate: float,
    action_std: torch.Tensor,
    rnd_weight: float | None,
    print_minimal: bool = False,
    width: int = 80,
    pad: int = 40,
  ) -> None:
    if self.writer is None:
      return

    collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
    iteration_time = collect_time + learn_time
    self.tot_timesteps += collection_size
    self.tot_time += iteration_time

    extras_string = ""
    terrain_ratio_values: dict[str, float] = {}
    terrain_level_values: dict[str, float] = {}
    terrain_level_max_values: dict[str, float] = {}
    if self.ep_extras:
      all_extra_keys = sorted(
        {
          key
          for ep_info in self.ep_extras
          for key in ep_info.keys()
        }
      )

      for key in all_extra_keys:
        infotensor = torch.tensor([], device=self.device)
        for ep_info in self.ep_extras:
          if key not in ep_info:
            continue
          if not isinstance(ep_info[key], torch.Tensor):
            ep_info[key] = torch.Tensor([ep_info[key]])
          if len(ep_info[key].shape) == 0:
            ep_info[key] = ep_info[key].unsqueeze(0)
          infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))

        if infotensor.numel() == 0:
          continue

        value = torch.mean(infotensor)
        value_float = float(value.detach().cpu().item())

        ratio_prefix = "Curriculum/terrain_ratio/"
        level_prefix = "Curriculum/terrain_level/"
        level_max_prefix = "Curriculum/terrain_level_max/"
        if key.startswith(ratio_prefix):
          terrain_name = key[len(ratio_prefix):]
          terrain_ratio_values[terrain_name] = value_float
        if key.startswith(level_max_prefix):
          terrain_name = key[len(level_max_prefix):]
          terrain_level_max_values[terrain_name] = value_float
        if key.startswith(level_prefix) and not key.startswith("Curriculum/terrain_level_max/"):
          terrain_name = key[len(level_prefix):]
          terrain_level_values[terrain_name] = value_float

        if "/" in key:
          self.writer.add_scalar(key, value, it)  # type: ignore
          if self._show_extra_in_cli(key):
            extras_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
        else:
          self.writer.add_scalar("Episode/" + key, value, it)  # type: ignore
          extras_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

      image_interval = int(self.cfg.get("curriculum_image_interval", 10))
      if image_interval <= 0:
        image_interval = 1

      if it % image_interval == 0:
        self._log_curriculum_bar_image(
          terrain_ratio_values,
          it,
          tag="CurriculumImages/terrain_ratio",
          title="Terrain spawn ratio",
          xlabel="spawn ratio [%]",
          xlim=(0.0, 100.0),
        )
        self._log_curriculum_bar_image(
          terrain_level_max_values,
          it,
          tag="CurriculumImages/terrain_level_max",
          title="Terrain level max",
          xlabel="max level",
          xlim=None,
        )
        self._log_curriculum_bar_image(
          terrain_level_values,
          it,
          tag="CurriculumImages/terrain_level",
          title="Terrain level mean",
          xlabel="mean level",
          xlim=None,
        )

    for key, value in loss_dict.items():
      self.writer.add_scalar(f"Loss/{key}", value, it)
    self.writer.add_scalar("Loss/learning_rate", learning_rate, it)
    self.writer.add_scalar("Policy/mean_std", action_std.mean().item(), it)

    fps = int(collection_size / (collect_time + learn_time))
    self.writer.add_scalar("Perf/total_fps", fps, it)
    self.writer.add_scalar("Perf/collection_time", collect_time, it)
    self.writer.add_scalar("Perf/learning_time", learn_time, it)

    if len(self.rewbuffer) > 0:
      if self.cfg["algorithm"]["rnd_cfg"]:
        self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(self.erewbuffer), it)
        self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(self.irewbuffer), it)
        self.writer.add_scalar("Rnd/weight", rnd_weight, it)  # type: ignore
      self.writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), it)
      self.writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), it)
      if self.logger_type != "WandbLogWriter":
        self.writer.add_scalar("Train/mean_reward/time", statistics.mean(self.rewbuffer), int(self.tot_time))
        self.writer.add_scalar(
          "Train/mean_episode_length/time",
          statistics.mean(self.lenbuffer),
          int(self.tot_time),
        )

    log_string = f"""{"#" * width}\n"""
    log_string += f"""\033[1m{f" Learning iteration {it}/{total_it} ".center(width)}\033[0m \n\n"""

    run_name = self.cfg.get("run_name")
    log_string += f"""{"Run name:":>{pad}} {run_name}\n""" if run_name else ""
    log_string += (
      f"""{"Total steps:":>{pad}} {self.tot_timesteps} \n"""
      f"""{"Steps per second:":>{pad}} {fps:.0f} \n"""
      f"""{"Collection time:":>{pad}} {collect_time:.3f}s \n"""
      f"""{"Learning time:":>{pad}} {learn_time:.3f}s \n"""
    )

    for key, value in loss_dict.items():
      log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""

    color_start, color_end, unlock_label = self._reward_color()
    roll_pitch_scale = self._get_roll_pitch_scale()
    if len(self.rewbuffer) > 0:
      if self.cfg["algorithm"]["rnd_cfg"]:
        log_string += f"""{"Mean extrinsic reward:":>{pad}} {statistics.mean(self.erewbuffer):.2f}\n"""
        log_string += f"""{"Mean intrinsic reward:":>{pad}} {statistics.mean(self.irewbuffer):.2f}\n"""
      mean_rew_field = f"{statistics.mean(self.rewbuffer):.2f}".rjust(6)
      log_string += f"""{"Mean reward:":>{pad}} {color_start}{mean_rew_field}{color_end}\n"""
      log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""

    log_string += f"""{"Uniform reset:":>{pad}} {color_start}{unlock_label}{color_end}\n"""
    if roll_pitch_scale is not None:
      log_string += f"""{"Roll/pitch scale:":>{pad}} {roll_pitch_scale:.3f}\n"""
    log_string += f"""{"Mean action std:":>{pad}} {action_std.mean().item():.2f}\n"""

    if not print_minimal:
      log_string += extras_string

    done_it = it + 1 - start_it
    remaining_it = total_it - start_it - done_it
    eta = self.tot_time / done_it * remaining_it
    log_string += (
      f"""{"-" * width}\n"""
      f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
      f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
      f"""{"ETA:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(eta))}\n"""
    )
    print(log_string)

    if self.logger_type == "WandbLogWriter":
      for video in pathlib.Path(self.log_dir).rglob("*.mp4"):  # type: ignore
        self.writer.save_video(video, it)  # type: ignore

    self.ep_extras.clear()
