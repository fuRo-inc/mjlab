from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg
from mjlab.utils import spec_config as spec_cfg


def _proportional_counts(num_envs: int, proportions: np.ndarray) -> np.ndarray:
  """Distribute *num_envs* across buckets proportionally.

  Every bucket gets at least one when ``num_envs >= len(proportions)``. Remaining slots
  are allocated via the Largest Remainder Method.
  """
  n = len(proportions)
  if num_envs >= n:
    counts = np.ones(n, dtype=int)
    remaining = num_envs - n
  else:
    counts = np.zeros(n, dtype=int)
    remaining = num_envs
  if remaining > 0:
    ideal = proportions * remaining
    floor = np.floor(ideal).astype(int)
    counts += floor
    leftover = remaining - floor.sum()
    if leftover > 0:
      order = np.argsort(-(ideal - floor))
      counts[order[:leftover]] += 1
  return counts


_DEFAULT_SUN_LIGHT = spec_cfg.LightCfg(
  name="sun", pos=(0.0, 0.0, 1.5), type="directional"
)

_DEFAULT_PLANE_TEXTURE = spec_cfg.TextureCfg(
  name="groundplane",
  type="2d",
  builtin="checker",
  mark="edge",
  rgb1=(0.2, 0.3, 0.4),
  rgb2=(0.1, 0.2, 0.3),
  markrgb=(0.8, 0.8, 0.8),
  width=300,
  height=300,
)

_DEFAULT_PLANE_MATERIAL = spec_cfg.MaterialCfg(
  name="groundplane",
  texuniform=True,
  texrepeat=(4.0, 4.0),
  reflectance=0.2,
  texture="groundplane",
  geom_names_expr=("terrain$",),
)


@dataclass
class TerrainEntityCfg(EntityCfg):
  """Configuration for terrain as an entity."""

  terrain_type: Literal["generator", "plane"] = "plane"
  """Type of terrain to generate. "generator" uses procedural terrain with
  sub-terrain grid, "plane" creates a flat ground plane."""
  terrain_generator: TerrainGeneratorCfg | None = None
  """Configuration for procedural terrain generation. Required when
  terrain_type is "generator"."""
  env_spacing: float | None = 2.0
  """Distance between environment origins when using grid layout. Required for
  "plane" terrain or when no sub-terrain origins exist."""
  max_init_terrain_level: int | None = None
  """Maximum initial difficulty level (row index) for environment placement in
  curriculum mode. None uses all available rows."""
  env_origin_sampling_mode: Literal["default", "uniform_cell"] = "default"
  """How env origins are assigned for generated terrain.

  ``"default"`` preserves the existing curriculum-style placement.
  ``"uniform_cell"`` stratifies reset batches over physical terrain cells.
  """
  num_envs: int = 1
  """Number of parallel environments to create. This will get overridden by the
  scene configuration if specified there."""
  textures: tuple[spec_cfg.TextureCfg, ...] = field(
    default_factory=lambda: (_DEFAULT_PLANE_TEXTURE,)
  )
  """Textures for the ground plane. Defaults to a checker pattern. Set to
  ``()`` to disable textures (e.g. when using ``dr.geom_rgba``)."""
  materials: tuple[spec_cfg.MaterialCfg, ...] = field(
    default_factory=lambda: (_DEFAULT_PLANE_MATERIAL,)
  )
  """Materials for the ground plane. Defaults to the checker material. Set to
  ``()`` to disable materials (e.g. when using ``dr.geom_rgba``)."""
  lights: tuple[spec_cfg.LightCfg, ...] = field(
    default_factory=lambda: (_DEFAULT_SUN_LIGHT,)
  )
  """Lights for the scene. Defaults to a directional sun light."""
  debug_vis: bool = False
  """Add visualization sites for environment origins, terrain origins, and
  flat patches. Defaults to False."""

  def build(self) -> TerrainEntity:
    raise TypeError(
      "TerrainEntityCfg.build() requires a device argument. "
      "Use TerrainEntity(cfg, device=...) directly."
    )


class TerrainEntity(Entity):
  """Terrain entity.

  The terrain is a grid of sub-terrain patches (num_rows x num_cols), each with
  a spawn origin. When num_envs exceeds the number of patches, environment
  origins are sampled from the sub-terrain origins.

  .. note::
    Environment allocation for procedural terrain: Columns (terrain types) are
    distributed across environments **by proportion** (matching each
    sub-terrain's ``proportion`` field) when a ``TerrainGeneratorCfg`` is
    available, or evenly when it is not.  Rows (difficulty levels) are randomly
    sampled.  This means multiple environments can spawn on the same (row, col)
    patch, leaving others unoccupied, even when num_envs > num_patches.

  See FAQ: "How does env_origins determine robot layout?"
  """

  cfg: TerrainEntityCfg

  def __init__(self, cfg: TerrainEntityCfg, device: str) -> None:
    self._device = device
    super().__init__(cfg)

  def _build_spec(self) -> None:
    self._spec = mujoco.MjSpec()

    if self.cfg.terrain_type == "generator":
      if self.cfg.terrain_generator is None:
        raise ValueError(
          "terrain_generator must be specified for terrain_type 'generator'"
        )
      terrain_generator_cls = getattr(
        self.cfg.terrain_generator,
        "class_type",
        TerrainGenerator,
      )
      if terrain_generator_cls is None:
        terrain_generator_cls = TerrainGenerator
      if terrain_generator_cls is not TerrainGenerator:
        print(
          f"[fuRo] Using terrain generator {terrain_generator_cls.__name__}"
        )
      terrain_generator = terrain_generator_cls(
        self.cfg.terrain_generator,
        device=self._device,
      )
      self._terrain_generator = terrain_generator
      terrain_generator.compile(self._spec)
      self.terrain_cell_difficulty_levels = self._cell_metadata_to_tensor(
        getattr(
          terrain_generator,
          "terrain_cell_difficulty_levels",
          None,
        )
      )
      self.terrain_cell_type_ids = self._cell_metadata_to_tensor(
        getattr(
          terrain_generator,
          "terrain_cell_type_ids",
          None,
        )
      )
      self.terrain_cell_logical_rows = self._cell_metadata_to_tensor(
        getattr(
          terrain_generator,
          "terrain_cell_logical_rows",
          None,
        )
      )
      self.terrain_cell_logical_cols = self._cell_metadata_to_tensor(
        getattr(
          terrain_generator,
          "terrain_cell_logical_cols",
          None,
        )
      )
      gen_cfg = self.cfg.terrain_generator
      proportions = np.array([s.proportion for s in gen_cfg.sub_terrains.values()])
      proportions = proportions / proportions.sum()
      self._configure_env_origins(terrain_generator.terrain_origins, proportions)
      self._flat_patches: dict[str, torch.Tensor] = {
        name: torch.from_numpy(arr).to(device=self._device, dtype=torch.float)
        for name, arr in terrain_generator.flat_patches.items()
      }
      self._flat_patch_radii: dict[str, float] = dict(
        terrain_generator.flat_patch_radii
      )
    elif self.cfg.terrain_type == "plane":
      self.terrain_cell_difficulty_levels = None
      self.terrain_cell_type_ids = None
      self.terrain_cell_logical_rows = None
      self.terrain_cell_logical_cols = None
      self._import_ground_plane("terrain")
      self._configure_env_origins()
      self._flat_patches: dict[str, torch.Tensor] = {}
      self._flat_patch_radii: dict[str, float] = {}
    else:
      raise ValueError(f"Unknown terrain type: {self.cfg.terrain_type}")

    if self.cfg.debug_vis:
      self._add_env_origin_sites()
      self._add_terrain_origin_sites()
      self._add_flat_patch_sites()

  def _add_initial_state_keyframe(self) -> None:
    pass  # No joints, no keyframe.

  # Terrain-specific properties.

  @property
  def flat_patches(self) -> dict[str, torch.Tensor]:
    return self._flat_patches

  @property
  def flat_patch_radii(self) -> dict[str, float]:
    return self._flat_patch_radii

  # Terrain origin management.

  def configure_env_origins(
    self,
    origins: np.ndarray | torch.Tensor | None = None,
    proportions: np.ndarray | None = None,
  ) -> None:
    """Configure the origins of the environments based on the terrain."""
    self._configure_env_origins(origins, proportions)

  def _configure_env_origins(
    self,
    origins: np.ndarray | torch.Tensor | None = None,
    proportions: np.ndarray | None = None,
  ) -> None:
    if origins is not None:
      if isinstance(origins, np.ndarray):
        origins = torch.from_numpy(origins)
      else:
        assert isinstance(origins, torch.Tensor)
      self.terrain_origins = origins.to(self._device, dtype=torch.float)
      if self.cfg.env_origin_sampling_mode == "default":
        self.env_origins = self._compute_env_origins_curriculum(
          self.cfg.num_envs, self.terrain_origins, proportions
        )
      elif self.cfg.env_origin_sampling_mode == "uniform_cell":
        self.env_origins = self._compute_env_origins_uniform_cell(
          self.cfg.num_envs, self.terrain_origins
        )
      else:
        raise ValueError(
          f"Unknown env_origin_sampling_mode: {self.cfg.env_origin_sampling_mode}"
        )
    else:
      self.terrain_origins = None
      if self.cfg.env_spacing is None:
        raise ValueError(
          "Environment spacing must be specified for configuring grid-like origins."
        )
      self.env_origins = self._compute_env_origins_grid(
        self.cfg.num_envs, self.cfg.env_spacing
      )

  def update_env_origins_before_reset(
    self, env_ids: torch.Tensor | slice | None
  ) -> None:
    """Update env origins just before reset events read them."""
    if self.terrain_origins is None:
      return
    if self.cfg.env_origin_sampling_mode != "uniform_cell":
      return
    env_ids = self._normalize_env_ids(env_ids)
    self._assign_uniform_cell_env_origins(env_ids, debug=True)

  def update_env_origins(
    self,
    env_ids: torch.Tensor,
    move_up: torch.Tensor,
    move_down: torch.Tensor,
  ) -> None:
    """Update the environment origins based on the terrain levels."""
    if self.terrain_origins is None:
      return
    assert self.env_origins is not None
    self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
    self.terrain_levels[env_ids] = torch.where(
      self.terrain_levels[env_ids] >= self.max_terrain_level,
      torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
      torch.clip(self.terrain_levels[env_ids], 0),
    )
    self.set_env_origins_from_logical(
      env_ids, self.terrain_levels[env_ids], self.terrain_types[env_ids]
    )

  def randomize_env_origins(self, env_ids: torch.Tensor) -> None:
    """Randomize the environment origins to random sub-terrains."""
    if self.terrain_origins is None:
      return
    assert self.env_origins is not None
    num_rows, num_cols = self.terrain_origins.shape[:2]
    num_envs = len(env_ids)
    physical_rows = torch.randint(
      0, num_rows, (num_envs,), device=self._device
    )
    physical_cols = torch.randint(
      0, num_cols, (num_envs,), device=self._device
    )
    self.set_env_origins_from_physical(env_ids, physical_rows, physical_cols)

  def set_env_origins_from_logical(
    self,
    env_ids: torch.Tensor,
    difficulty_levels: torch.Tensor,
    type_ids: torch.Tensor,
  ) -> None:
    """Set origins from logical difficulty/type indices."""
    if self.terrain_origins is None:
      return
    physical_rows, physical_cols = self._resolve_physical_cell_indices(
      difficulty_levels, type_ids
    )
    self.set_env_origins_from_physical(
      env_ids,
      physical_rows,
      physical_cols,
      difficulty_levels=difficulty_levels,
      type_ids=type_ids,
    )

  def set_env_origins_from_physical(
    self,
    env_ids: torch.Tensor,
    physical_rows: torch.Tensor,
    physical_cols: torch.Tensor,
    *,
    difficulty_levels: torch.Tensor | None = None,
    type_ids: torch.Tensor | None = None,
    debug: bool = False,
  ) -> None:
    """Set origins from physical terrain cell indices."""
    if self.terrain_origins is None:
      return
    assert self.env_origins is not None
    env_ids = self._normalize_env_ids(env_ids)
    physical_rows = physical_rows.to(device=self._device, dtype=torch.long).clone()
    physical_cols = physical_cols.to(device=self._device, dtype=torch.long).clone()
    if difficulty_levels is None or type_ids is None:
      difficulty_levels, type_ids = self._resolve_logical_cell_indices(
        physical_rows, physical_cols
      )
      difficulty_levels = difficulty_levels.clone()
      type_ids = type_ids.clone()
    else:
      difficulty_levels = difficulty_levels.to(
        device=self._device, dtype=torch.long
      ).clone()
      type_ids = type_ids.to(device=self._device, dtype=torch.long).clone()

    self._ensure_cell_index_buffers()
    self.terrain_physical_rows[env_ids] = physical_rows
    self.terrain_physical_cols[env_ids] = physical_cols
    self.terrain_levels[env_ids] = difficulty_levels
    self.terrain_types[env_ids] = type_ids
    self.env_origins[env_ids] = self.terrain_origins[physical_rows, physical_cols]

    # if debug:
    #   print("[fuRo] uniform_cell terrain origin sampling enabled")
    #   print(
    #     "physical_rows min/max:",
    #     physical_rows.min().item(),
    #     physical_rows.max().item(),
    #   )
    #   print(
    #     "physical_cols min/max:",
    #     physical_cols.min().item(),
    #     physical_cols.max().item(),
    #   )
    #   print(
    #     "difficulty_levels min/max:",
    #     difficulty_levels.min().item(),
    #     difficulty_levels.max().item(),
    #   )
    #   print("type_ids min/max:", type_ids.min().item(), type_ids.max().item())

  def get_env_physical_cell_indices(
    self, env_ids: torch.Tensor | slice | None = None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return physical row/col indices for env-indexed terrain arrays."""
    env_ids = self._normalize_env_ids(env_ids)
    if hasattr(self, "terrain_physical_rows") and hasattr(
      self, "terrain_physical_cols"
    ):
      return self.terrain_physical_rows[env_ids], self.terrain_physical_cols[env_ids]
    return self.terrain_levels[env_ids], self.terrain_types[env_ids]

  # Private methods.

  def _cell_metadata_to_tensor(
    self, value: np.ndarray | torch.Tensor | None
  ) -> torch.Tensor | None:
    if value is None:
      return None
    if isinstance(value, np.ndarray):
      value = torch.from_numpy(value)
    else:
      assert isinstance(value, torch.Tensor)
    return value.to(self._device, dtype=torch.long)

  def _normalize_env_ids(
    self, env_ids: torch.Tensor | slice | None
  ) -> torch.Tensor:
    all_env_ids = torch.arange(self.cfg.num_envs, device=self._device, dtype=torch.long)
    if env_ids is None:
      return all_env_ids
    if isinstance(env_ids, slice):
      return all_env_ids[env_ids]
    return env_ids.to(device=self._device, dtype=torch.long)

  def _ensure_cell_index_buffers(self) -> None:
    has_buffers = (
      hasattr(self, "terrain_levels")
      and hasattr(self, "terrain_types")
      and hasattr(self, "terrain_physical_rows")
      and hasattr(self, "terrain_physical_cols")
      and self.terrain_levels.shape == (self.cfg.num_envs,)
      and self.terrain_types.shape == (self.cfg.num_envs,)
      and self.terrain_physical_rows.shape == (self.cfg.num_envs,)
      and self.terrain_physical_cols.shape == (self.cfg.num_envs,)
    )
    if has_buffers:
      return
    self.terrain_levels = torch.zeros(
      self.cfg.num_envs, device=self._device, dtype=torch.long
    )
    self.terrain_types = torch.zeros(
      self.cfg.num_envs, device=self._device, dtype=torch.long
    )
    self.terrain_physical_rows = torch.zeros(
      self.cfg.num_envs, device=self._device, dtype=torch.long
    )
    self.terrain_physical_cols = torch.zeros(
      self.cfg.num_envs, device=self._device, dtype=torch.long
    )

  def _resolve_logical_cell_indices(
    self, physical_rows: torch.Tensor, physical_cols: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    difficulty_levels = self.terrain_cell_difficulty_levels
    type_ids = self.terrain_cell_type_ids
    if difficulty_levels is None or type_ids is None:
      return physical_rows.long(), physical_cols.long()
    return (
      difficulty_levels[physical_rows, physical_cols].long(),
      type_ids[physical_rows, physical_cols].long(),
    )

  def _resolve_physical_cell_indices(
    self, difficulty_levels: torch.Tensor, type_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    cell_difficulty = self.terrain_cell_difficulty_levels
    cell_types = self.terrain_cell_type_ids
    if cell_difficulty is None or cell_types is None:
      return difficulty_levels.long(), type_ids.long()

    num_rows, num_cols = cell_difficulty.shape
    flat_difficulty = cell_difficulty.reshape(-1)
    flat_types = cell_types.reshape(-1)
    type_factor = int(flat_types.max().item()) + 1
    flat_keys = flat_difficulty * type_factor + flat_types
    keys = difficulty_levels.long() * type_factor + type_ids.long()
    matches = keys[:, None] == flat_keys[None, :]
    if not torch.all(matches.any(dim=1)):
      missing = keys[~matches.any(dim=1)].detach().cpu().tolist()
      raise ValueError(f"Logical terrain cells not found in metadata: {missing}")
    flat_ids = matches.to(torch.long).argmax(dim=1)
    return flat_ids // num_cols, flat_ids % num_cols

  def _import_ground_plane(self, name: str) -> None:
    self._spec.worldbody.add_body(name=name).add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=(0, 0, 0.01),
    )

  def _add_env_origin_sites(self) -> None:
    if self.env_origins is None:
      return
    origin_site_radius: float = 0.3
    origin_site_color = (0.2, 0.6, 0.2, 0.3)
    if isinstance(self.env_origins, torch.Tensor):
      env_origins_np = self.env_origins.cpu().numpy()
    else:
      env_origins_np = self.env_origins
    for env_id, origin in enumerate(env_origins_np):
      self._spec.worldbody.add_site(
        name=f"env_origin_{env_id}",
        pos=origin,
        size=(origin_site_radius,) * 3,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        rgba=origin_site_color,
        group=4,
      )

  def _add_terrain_origin_sites(self) -> None:
    if self.terrain_origins is None:
      return
    if isinstance(self.terrain_origins, torch.Tensor):
      terrain_origins_np = self.terrain_origins.cpu().numpy()
    else:
      terrain_origins_np = self.terrain_origins
    terrain_origin_site_radius: float = 0.5
    terrain_origin_site_color = (0.2, 0.2, 0.6, 0.3)
    num_rows, num_cols = terrain_origins_np.shape[:2]
    for row in range(num_rows):
      for col in range(num_cols):
        origin = terrain_origins_np[row, col]
        self._spec.worldbody.add_site(
          name=f"terrain_origin_{row}_{col}",
          pos=origin,
          size=(terrain_origin_site_radius,) * 3,
          type=mujoco.mjtGeom.mjGEOM_SPHERE,
          rgba=terrain_origin_site_color,
          group=5,
        )

  def _add_flat_patch_sites(self) -> None:
    if not self._flat_patches:
      return
    site_thickness = 0.02
    site_color = (0.9, 0.6, 0.1, 0.1)
    for name, patches_tensor in self._flat_patches.items():
      radius = self._flat_patch_radii.get(name, 0.5)
      patches_np = patches_tensor.cpu().numpy()
      num_rows, num_cols, num_patches, _ = patches_np.shape
      for row in range(num_rows):
        for col in range(num_cols):
          for p in range(num_patches):
            pos = patches_np[row, col, p]
            self._spec.worldbody.add_site(
              name=f"flat_patch_{name}_{row}_{col}_{p}",
              pos=pos,
              size=(radius, radius, site_thickness),
              type=mujoco.mjtGeom.mjGEOM_BOX,
              rgba=site_color,
              group=3,
            )

  def _compute_env_origins_curriculum(
    self,
    num_envs: int,
    origins: torch.Tensor,
    proportions: np.ndarray | None = None,
  ) -> torch.Tensor:
    """Compute env origins from sub-terrain origins.

    Args:
      num_envs: Number of environments to place.
      origins: Sub-terrain origins, shape ``[num_rows, num_cols, 3]``.
      proportions: Normalized per-column weights. When provided, robots
        are distributed proportionally (every column gets at least one
        when ``num_envs >= num_cols``). ``None`` gives even distribution.
    """
    num_rows, num_cols = origins.shape[:2]
    if self.cfg.max_init_terrain_level is None:
      max_init_level = num_rows - 1
    else:
      max_init_level = min(self.cfg.max_init_terrain_level, num_rows - 1)
    self.max_terrain_level = self._logical_num_difficulty_levels(num_rows)
    physical_rows = torch.randint(
      0, max_init_level + 1, (num_envs,), device=self._device
    )

    if proportions is not None and len(proportions) == num_cols:
      counts = _proportional_counts(num_envs, proportions)
      physical_cols = torch.repeat_interleave(
        torch.arange(num_cols, device=self._device),
        torch.from_numpy(counts).to(self._device),
      )
    else:
      physical_cols = torch.div(
        torch.arange(num_envs, device=self._device),
        (num_envs / num_cols),
        rounding_mode="floor",
      ).to(torch.long)

    self.terrain_physical_rows = physical_rows.long()
    self.terrain_physical_cols = physical_cols.long()
    self.terrain_levels, self.terrain_types = self._resolve_logical_cell_indices(
      self.terrain_physical_rows, self.terrain_physical_cols
    )
    env_origins = torch.zeros(num_envs, 3, device=self._device)
    env_origins[:] = origins[self.terrain_physical_rows, self.terrain_physical_cols]
    return env_origins

  def _compute_env_origins_uniform_cell(
    self, num_envs: int, origins: torch.Tensor
  ) -> torch.Tensor:
    num_rows, _ = origins.shape[:2]
    self.max_terrain_level = self._logical_num_difficulty_levels(num_rows)
    self._ensure_cell_index_buffers()
    self.env_origins = torch.zeros(num_envs, 3, device=self._device)
    env_ids = torch.arange(num_envs, device=self._device, dtype=torch.long)
    self._assign_uniform_cell_env_origins(env_ids, debug=True)
    return self.env_origins

  def _assign_uniform_cell_env_origins(
    self, env_ids: torch.Tensor, *, debug: bool
  ) -> None:
    if self.terrain_origins is None or env_ids.numel() == 0:
      return
    num_rows, num_cols = self.terrain_origins.shape[:2]
    num_cells = num_rows * num_cols
    n = env_ids.numel()
    offset = torch.randint(0, num_cells, (1,), device=self._device)
    cell_ids = (torch.arange(n, device=self._device) + offset) % num_cells
    cell_ids = cell_ids[torch.randperm(n, device=self._device)]
    physical_rows = torch.div(cell_ids, num_cols, rounding_mode="floor")
    physical_cols = cell_ids % num_cols
    self.set_env_origins_from_physical(
      env_ids,
      physical_rows,
      physical_cols,
      debug=debug,
    )

  def _logical_num_difficulty_levels(self, fallback_num_rows: int) -> int:
    difficulty_levels = self.terrain_cell_difficulty_levels
    if difficulty_levels is None:
      return fallback_num_rows
    return int(difficulty_levels.max().item()) + 1

  def _compute_env_origins_grid(
    self, num_envs: int, env_spacing: float
  ) -> torch.Tensor:
    env_origins = torch.zeros(num_envs, 3, device=self._device)
    num_rows = np.ceil(num_envs / int(np.sqrt(num_envs)))
    num_cols = np.ceil(num_envs / num_rows)
    ii, jj = torch.meshgrid(
      torch.arange(num_rows, device=self._device),
      torch.arange(num_cols, device=self._device),
      indexing="ij",
    )
    env_origins[:, 0] = -(ii.flatten()[:num_envs] - (num_rows - 1) / 2) * env_spacing
    env_origins[:, 1] = (jj.flatten()[:num_envs] - (num_cols - 1) / 2) * env_spacing
    env_origins[:, 2] = 0.0
    return env_origins
