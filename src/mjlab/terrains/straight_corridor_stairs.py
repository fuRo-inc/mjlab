"""Lightweight straight stairs constrained by two continuous side walls."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.utils.color import brand_ramp, darken_rgba

_MUJOCO_PURPLE = (0.58, 0.36, 0.90)
_BARRIER_COLOR = (0.90, 0.45, 0.20, 1.0)
_MIN_GEOM_SIZE = 1.0e-6
_MIN_FLOOR_HEIGHT = 0.05


@dataclass(kw_only=True)
class BoxStraightCorridorStairsTerrainCfg(SubTerrainCfg):
  """One straight staircase with low/high landings and two continuous walls.

  The staircase always rises along local +X. The whole terrain cell is covered
  by one flat low floor, while ``border_width`` is used only as a placement
  margin for the staircase and walls. Compared with a four-sided pyramid
  corridor, this terrain uses only ``num_steps + 4`` collision boxes.
  """

  border_width: float = 0.0
  """Placement margin between the staircase assembly and the cell edge."""

  step_height_range: tuple[float, float]
  """Minimum and maximum riser height, interpolated by difficulty."""

  step_width: float = 0.3
  """Fallback tread depth when ``step_width_range`` is not set."""

  step_width_range: tuple[float, float] | None = None
  """Optional tread-depth range, interpolated from max to min by difficulty."""

  corridor_width_range: tuple[float, float] = (0.45, 0.70)
  """Minimum and maximum clear width between the two walls."""

  landing_length: float = 1.3
  """Length of both low and high landings along the staircase axis."""

  wall_height_range: tuple[float, float] = (0.40, 0.60)
  """Wall height above the high landing, interpolated by difficulty."""

  wall_thickness: float = 0.20
  """Thickness of each continuous side wall."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng

    if self.border_width < 0.0:
      raise ValueError("border_width must be non-negative")
    if self.landing_length <= 0.0:
      raise ValueError("landing_length must be positive")
    if self.wall_thickness <= 0.0:
      raise ValueError("wall_thickness must be positive")
    if self.step_height_range[0] < 0.0:
      raise ValueError("step_height_range minimum must be non-negative")
    if self.step_height_range[1] < self.step_height_range[0]:
      raise ValueError("step_height_range must satisfy min <= max")
    if self.corridor_width_range[0] <= 0.0:
      raise ValueError("corridor_width_range minimum must be positive")
    if self.corridor_width_range[1] < self.corridor_width_range[0]:
      raise ValueError("corridor_width_range must satisfy min <= max")
    if self.wall_height_range[0] < 0.0:
      raise ValueError("wall_height_range minimum must be non-negative")
    if self.wall_height_range[1] < self.wall_height_range[0]:
      raise ValueError("wall_height_range must satisfy min <= max")

    step_height = self.step_height_range[0] + difficulty * (
      self.step_height_range[1] - self.step_height_range[0]
    )
    step_width = self.step_width
    if self.step_width_range is not None:
      step_width = self.step_width_range[1] - difficulty * (
        self.step_width_range[1] - self.step_width_range[0]
      )
    if step_width <= 0.0:
      raise ValueError("step width must be positive")

    corridor_width = self.corridor_width_range[1] - difficulty * (
      self.corridor_width_range[1] - self.corridor_width_range[0]
    )
    wall_height = self.wall_height_range[0] + difficulty * (
      self.wall_height_range[1] - self.wall_height_range[0]
    )

    terrain_center = np.array([0.5 * self.size[0], 0.5 * self.size[1], 0.0])
    interior_x = self.size[0] - 2.0 * self.border_width
    interior_y = self.size[1] - 2.0 * self.border_width
    if interior_x <= 2.0 * self.landing_length:
      raise ValueError("terrain interior is too short for both landings")
    if corridor_width + 2.0 * self.wall_thickness > interior_y:
      raise ValueError("corridor and walls do not fit inside terrain width")

    available_stair_length = interior_x - 2.0 * self.landing_length
    num_steps = int(available_stair_length / step_width)
    if num_steps < 1:
      raise ValueError("terrain interior must fit at least one stair")
    stair_length = num_steps * step_width
    used_length = 2.0 * self.landing_length + stair_length
    low_edge_x = terrain_center[0] - 0.5 * used_length
    stair_start_x = low_edge_x + self.landing_length
    stair_end_x = stair_start_x + stair_length
    high_edge_x = stair_end_x + self.landing_length
    total_rise = num_steps * step_height

    body = spec.body("terrain")
    boxes: list[mujoco.MjsGeom] = []
    colors: list[tuple[float, float, float, float]] = []

    def add_box(
      size_xyz: tuple[float, float, float],
      pos_xyz: tuple[float, float, float],
      color: tuple[float, float, float, float],
    ) -> None:
      safe_size = tuple(max(_MIN_GEOM_SIZE, 0.5 * dim) for dim in size_xyz)
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=safe_size,
        pos=pos_xyz,
        contype=1,
        conaffinity=0,
      )
      boxes.append(geom)
      colors.append(color)

    # Terrain geoms need to collide with robot geoms, but not with each other.
    # Using contype=1 and conaffinity=0 preserves robot-terrain contacts when the
    # robot uses the default collision mask, while filtering floor-step,
    # step-step, landing-step, and wall-floor self contacts.
    floor_color = darken_rgba(brand_ramp(_MUJOCO_PURPLE, 0.0), 0.85)
    floor_height = _MIN_FLOOR_HEIGHT
    add_box(
      (self.size[0], self.size[1], floor_height),
      (terrain_center[0], terrain_center[1], -0.5 * floor_height),
      floor_color,
    )

    for k in range(num_steps):
      top_z = (k + 1) * step_height
      x_pos = stair_start_x + (k + 0.5) * step_width
      step_color = brand_ramp(_MUJOCO_PURPLE, k / max(num_steps - 1, 1))
      add_box(
        (step_width, corridor_width, top_z),
        (x_pos, terrain_center[1], 0.5 * top_z),
        step_color,
      )

    add_box(
      (self.landing_length, corridor_width, total_rise),
      (
        stair_end_x + 0.5 * self.landing_length,
        terrain_center[1],
        0.5 * total_rise,
      ),
      brand_ramp(_MUJOCO_PURPLE, 1.0),
    )

    if wall_height > _MIN_GEOM_SIZE:
      wall_total_height = total_rise + wall_height
      wall_center_z = 0.5 * wall_total_height
      wall_length = high_edge_x - low_edge_x
      wall_center_x = 0.5 * (low_edge_x + high_edge_x)
      half_corridor = 0.5 * corridor_width
      for y_pos in (
        terrain_center[1] - half_corridor - 0.5 * self.wall_thickness,
        terrain_center[1] + half_corridor + 0.5 * self.wall_thickness,
      ):
        add_box(
          (wall_length, self.wall_thickness, wall_total_height),
          (wall_center_x, y_pos, wall_center_z),
          _BARRIER_COLOR,
        )

    origin = np.array([terrain_center[0], terrain_center[1], 0.0])
    geometries = [
      TerrainGeometry(geom=geom, color=color)
      for geom, color in zip(boxes, colors, strict=True)
    ]
    return TerrainOutput(origin=origin, geometries=geometries)
