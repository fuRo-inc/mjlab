"""Pyramid stairs constrained by continuous side walls.

Each stair face contains one central corridor. Two stepped side walls follow the
radial direction between the outer edge and center platform. The same terrain
can be generated as a normal pyramid (high center, downhill outward) or an
inverted pyramid (low center, uphill outward).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_border
from mjlab.utils.color import brand_ramp, darken_rgba

_MUJOCO_PURPLE = (0.58, 0.36, 0.90)
_BARRIER_COLOR = (0.90, 0.45, 0.20, 1.0)
_MIN_BORDER_HEIGHT = 0.05
_MIN_GEOM_SIZE = 1.0e-6


@dataclass(kw_only=True)
class BoxCorridorPyramidStairsTerrainCfg(SubTerrainCfg):
  """Pyramid stairs with a narrow, wall-bounded corridor on every face."""

  border_width: float = 0.0
  """Width of the flat border frame around the staircase, in meters."""

  step_height_range: tuple[float, float]
  """Min and max step height, interpolated by difficulty."""

  step_width: float = 0.3
  """Fallback tread depth when ``step_width_range`` is not set."""

  step_width_range: tuple[float, float] | None = None
  """Optional tread-depth range, interpolated from max to min by difficulty."""

  platform_width: float = 1.0
  """Side length of the center platform."""

  corridor_width_range: tuple[float, float] = (0.70, 1.00)
  """Min and max clear corridor width."""

  wall_height_range: tuple[float, float] = (0.10, 0.25)
  """Min and max wall height above each tread, interpolated by difficulty."""

  wall_thickness: float = 0.05
  """Side-wall thickness, in meters."""

  guard_center_platform: bool = True
  """Close the platform perimeter except at the four corridor openings."""

  inverted: bool = False
  """If True, make the center platform low and the outer stairs high.

  The robot can then spawn on the center platform and climb outward. If False,
  the center platform is high and the robot descends outward.
  """

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng

    if self.border_width < 0.0:
      raise ValueError("border_width must be non-negative")
    if self.platform_width <= 0.0:
      raise ValueError("platform_width must be positive")
    if self.wall_thickness <= 0.0:
      raise ValueError("wall_thickness must be positive")
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
    if step_height < 0.0:
      raise ValueError("step height must be non-negative")
    if step_width <= 0.0:
      raise ValueError("step width must be positive")

    corridor_width = self.corridor_width_range[1] - difficulty * (
      self.corridor_width_range[1] - self.corridor_width_range[0]
    )
    wall_height = self.wall_height_range[0] + difficulty * (
      self.wall_height_range[1] - self.wall_height_range[0]
    )

    if corridor_width + 2.0 * self.wall_thickness > self.platform_width:
      raise ValueError(
        "corridor width plus both walls must fit within platform_width"
      )

    body = spec.body("terrain")
    boxes: list[mujoco.MjsGeom] = []
    colors: list[tuple[float, float, float, float]] = []

    terrain_center = np.array([0.5 * self.size[0], 0.5 * self.size[1], 0.0])
    terrain_size = np.array(
      [self.size[0] - 2.0 * self.border_width, self.size[1] - 2.0 * self.border_width]
    )
    if np.any(terrain_size <= self.platform_width):
      raise ValueError("terrain interior must be larger than platform_width")

    num_steps_x = int((terrain_size[0] - self.platform_width) / (2.0 * step_width))
    num_steps_y = int((terrain_size[1] - self.platform_width) / (2.0 * step_width))
    num_steps = max(0, min(num_steps_x, num_steps_y))

    first_color = brand_ramp(_MUJOCO_PURPLE, 0.0)
    border_color = darken_rgba(first_color, 0.85)
    if self.border_width > 0.0:
      border_height = max(step_height, _MIN_BORDER_HEIGHT)
      border_center = (
        terrain_center[0],
        terrain_center[1],
        -0.5 * border_height,
      )
      border_boxes = make_border(
        body,
        self.size,
        tuple(terrain_size),
        border_height,
        border_center,
      )
      boxes.extend(border_boxes)
      colors.extend([border_color] * len(border_boxes))

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
      )
      boxes.append(geom)
      colors.append(color)

    def tread_top(k: int) -> float:
      if self.inverted:
        return terrain_center[2] + (num_steps - k) * step_height
      return terrain_center[2] + (k + 1) * step_height

    def support_box_z_and_height(top_z: float) -> tuple[float, float]:
      bottom_z = terrain_center[2] - step_height
      height = max(_MIN_GEOM_SIZE, top_z - bottom_z)
      return 0.5 * (top_z + bottom_z), height

    half_corridor = 0.5 * corridor_width

    for k in range(num_steps):
      color_t = k / max(num_steps - 1, 1)
      if self.inverted:
        color_t = 1.0 - color_t
      step_color = brand_ramp(_MUJOCO_PURPLE, color_t)
      remaining_x = terrain_size[0] - 2.0 * k * step_width
      remaining_y = terrain_size[1] - 2.0 * k * step_width
      top_z = tread_top(k)
      box_z, box_height = support_box_z_and_height(top_z)
      nominal_offset = (k + 0.5) * step_width

      top_y = terrain_center[1] + 0.5 * terrain_size[1] - nominal_offset
      bottom_y = terrain_center[1] - 0.5 * terrain_size[1] + nominal_offset
      right_x = terrain_center[0] + 0.5 * terrain_size[0] - nominal_offset
      left_x = terrain_center[0] - 0.5 * terrain_size[0] + nominal_offset

      add_box(
        (remaining_x, step_width, box_height),
        (terrain_center[0], top_y, box_z),
        step_color,
      )
      add_box(
        (remaining_x, step_width, box_height),
        (terrain_center[0], bottom_y, box_z),
        step_color,
      )

      side_span_y = max(_MIN_GEOM_SIZE, remaining_y - 2.0 * step_width)
      add_box(
        (step_width, side_span_y, box_height),
        (right_x, terrain_center[1], box_z),
        step_color,
      )
      add_box(
        (step_width, side_span_y, box_height),
        (left_x, terrain_center[1], box_z),
        step_color,
      )

      if wall_height <= _MIN_GEOM_SIZE:
        continue

      wall_z = top_z + 0.5 * wall_height

      for x_pos in (
        terrain_center[0] - half_corridor - 0.5 * self.wall_thickness,
        terrain_center[0] + half_corridor + 0.5 * self.wall_thickness,
      ):
        add_box(
          (self.wall_thickness, step_width, wall_height),
          (x_pos, top_y, wall_z),
          _BARRIER_COLOR,
        )
        add_box(
          (self.wall_thickness, step_width, wall_height),
          (x_pos, bottom_y, wall_z),
          _BARRIER_COLOR,
        )

      for y_pos in (
        terrain_center[1] - half_corridor - 0.5 * self.wall_thickness,
        terrain_center[1] + half_corridor + 0.5 * self.wall_thickness,
      ):
        add_box(
          (step_width, self.wall_thickness, wall_height),
          (right_x, y_pos, wall_z),
          _BARRIER_COLOR,
        )
        add_box(
          (step_width, self.wall_thickness, wall_height),
          (left_x, y_pos, wall_z),
          _BARRIER_COLOR,
        )

    center_top_z = (
      terrain_center[2]
      if self.inverted
      else terrain_center[2] + (num_steps + 1) * step_height
    )
    center_box_z, center_height = support_box_z_and_height(center_top_z)
    center_dims = (
      terrain_size[0] - 2.0 * num_steps * step_width,
      terrain_size[1] - 2.0 * num_steps * step_width,
      center_height,
    )
    center_pos = (
      terrain_center[0],
      terrain_center[1],
      center_box_z,
    )
    center_color = brand_ramp(_MUJOCO_PURPLE, 0.0 if self.inverted else 1.0)
    add_box(center_dims, center_pos, center_color)

    if self.guard_center_platform and wall_height > _MIN_GEOM_SIZE:
      platform_top_z = center_top_z
      wall_z = platform_top_z + 0.5 * wall_height
      half_x = 0.5 * center_dims[0]
      half_y = 0.5 * center_dims[1]
      segment_x = 0.5 * (center_dims[0] - corridor_width)
      segment_y = 0.5 * (center_dims[1] - corridor_width)

      if segment_x > _MIN_GEOM_SIZE:
        x_offset = half_corridor + 0.5 * segment_x
        for y_pos in (
          terrain_center[1] - half_y + 0.5 * self.wall_thickness,
          terrain_center[1] + half_y - 0.5 * self.wall_thickness,
        ):
          for x_pos in (
            terrain_center[0] - x_offset,
            terrain_center[0] + x_offset,
          ):
            add_box(
              (segment_x, self.wall_thickness, wall_height),
              (x_pos, y_pos, wall_z),
              _BARRIER_COLOR,
            )

      if segment_y > _MIN_GEOM_SIZE:
        y_offset = half_corridor + 0.5 * segment_y
        for x_pos in (
          terrain_center[0] - half_x + 0.5 * self.wall_thickness,
          terrain_center[0] + half_x - 0.5 * self.wall_thickness,
        ):
          for y_pos in (
            terrain_center[1] - y_offset,
            terrain_center[1] + y_offset,
          ):
            add_box(
              (self.wall_thickness, segment_y, wall_height),
              (x_pos, y_pos, wall_z),
              _BARRIER_COLOR,
            )

    origin = np.array([terrain_center[0], terrain_center[1], center_top_z])
    geometries = [
      TerrainGeometry(geom=geom, color=color)
      for geom, color in zip(boxes, colors, strict=True)
    ]
    return TerrainOutput(origin=origin, geometries=geometries)
