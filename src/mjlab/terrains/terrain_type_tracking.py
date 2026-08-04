"""Track the actual generated sub-terrain type for every terrain cell.

This keeps terrain-name lookup correct in both curriculum and random generation
modes. ``TerrainEntity.terrain_types`` remains the logical column assignment;
``generated_terrain_types`` stores the actual sub-terrain index generated in
that physical cell.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from mjlab.terrains.terrain_entity import TerrainEntity
from mjlab.terrains.terrain_generator import TerrainGenerator

_ORIGINAL_GENERATOR_INIT = TerrainGenerator.__init__
_ORIGINAL_CREATE_TERRAIN_GEOM = TerrainGenerator._create_terrain_geom
_ORIGINAL_CONFIGURE_ENV_ORIGINS = TerrainEntity._configure_env_origins

# TerrainGenerator passes its terrain_origins ndarray directly to TerrainEntity.
# Keep the generated-type grid associated with that exact ndarray until the
# entity consumes it during construction.
_GENERATED_TYPE_REGISTRY: dict[int, np.ndarray] = {}


def _generator_init_with_type_grid(self: TerrainGenerator, *args: Any, **kwargs: Any) -> None:
  _ORIGINAL_GENERATOR_INIT(self, *args, **kwargs)
  self.generated_terrain_types = np.full(
    (self.cfg.num_rows, self._num_cols),
    -1,
    dtype=np.int64,
  )
  _GENERATED_TYPE_REGISTRY[id(self.terrain_origins)] = self.generated_terrain_types


def _create_terrain_geom_with_type_tracking(
  self: TerrainGenerator,
  spec,
  world_position: np.ndarray,
  difficulty: float,
  cfg,
  sub_row: int,
  sub_col: int,
) -> np.ndarray:
  sub_terrain_cfgs = list(self.cfg.sub_terrains.values())
  try:
    generated_type = next(
      index for index, sub_cfg in enumerate(sub_terrain_cfgs) if sub_cfg is cfg
    )
  except StopIteration as exc:
    raise RuntimeError(
      "Generated sub-terrain config is not present in TerrainGeneratorCfg.sub_terrains."
    ) from exc

  self.generated_terrain_types[sub_row, sub_col] = generated_type
  return _ORIGINAL_CREATE_TERRAIN_GEOM(
    self,
    spec,
    world_position,
    difficulty,
    cfg,
    sub_row,
    sub_col,
  )


def _configure_env_origins_with_generated_types(
  self: TerrainEntity,
  origins: np.ndarray | torch.Tensor | None = None,
  proportions: np.ndarray | None = None,
) -> None:
  generated_types_np = None
  if isinstance(origins, np.ndarray):
    generated_types_np = _GENERATED_TYPE_REGISTRY.get(id(origins))

  _ORIGINAL_CONFIGURE_ENV_ORIGINS(self, origins, proportions)

  if origins is None:
    self.generated_terrain_types = None
    return

  if generated_types_np is None:
    # Backward-compatible fallback for custom generators that do not use the
    # standard TerrainGenerator path. In curriculum mode physical columns are
    # terrain types; otherwise the actual type is unknown.
    terrain_generator_cfg = self.cfg.terrain_generator
    if terrain_generator_cfg is not None and terrain_generator_cfg.curriculum:
      num_rows, num_cols = self.terrain_origins.shape[:2]
      generated_types = torch.arange(
        num_cols,
        device=self._device,
        dtype=torch.long,
      ).unsqueeze(0).expand(num_rows, -1).clone()
      self.generated_terrain_types = generated_types
    else:
      self.generated_terrain_types = None
    return

  self.generated_terrain_types = torch.from_numpy(generated_types_np).to(
    device=self._device,
    dtype=torch.long,
  )
  _GENERATED_TYPE_REGISTRY.pop(id(origins), None)


def _get_env_terrain_types(self: TerrainEntity) -> torch.Tensor:
  """Return the actual generated sub-terrain index for every environment."""
  generated_types = getattr(self, "generated_terrain_types", None)
  if generated_types is None:
    raise RuntimeError(
      "Actual generated terrain types are unavailable for this terrain entity."
    )
  if self.terrain_origins is None:
    raise RuntimeError("Procedural terrain origins are unavailable.")
  return generated_types[self.terrain_levels, self.terrain_types]


def install_generated_terrain_type_tracking() -> None:
  """Install terrain-type tracking once."""
  if getattr(TerrainGenerator, "_generated_type_tracking_installed", False):
    return

  TerrainGenerator.__init__ = _generator_init_with_type_grid
  TerrainGenerator._create_terrain_geom = _create_terrain_geom_with_type_tracking
  TerrainEntity._configure_env_origins = _configure_env_origins_with_generated_types
  TerrainEntity.get_env_terrain_types = _get_env_terrain_types
  TerrainGenerator._generated_type_tracking_installed = True


install_generated_terrain_type_tracking()
