"""Pyramid stairs constrained by continuous side walls.

Each stair face contains one central corridor. Two stepped side walls follow the
climbing direction from the outer edge to the center platform, limiting lateral
escape and large body-yaw deviations while preserving the ordinary stair treads.
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
  """Min and max clear corridor width.

  Width is interpolated from max to min as difficulty increases, so higher
  difficulty provides less lateral room.
  """

  wall_height_range: tuple[float, float] = (0.10, 0.25)
  """Min and max wall height above each tread, interpolated by difficulty."""

  wall_thickness: float = 0.05
  """Side-wall thickness, in meters."""

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

    half_corridor = 0.5 * corridor_width

    for k in range(num_steps):
      step_color = brand_ramp(_MUJOCO_PURPLE, k / max(num_steps - 1, 1))
      remaining_x = terrain_size[0] - 2.0 * k * step_width
      remaining_y = terrain_size[1] - 2.0 * k * step_width
      box_z = terrain_center[2] + 0.5 * k * step_height
      box_height = (k + 2) * step_height
      nominal_offset = (k + 0.5) * step_width

      top_y = terrain_center[1] + 0.5 * terrain_size[1] - nominal_offset
      bottom_y = terrain_center[1] - 0.5 * terrain_size[1] + nominal_offset
      right_x = terrain_center[0] + 0.5 * terrain_size[0] - nominal_offset
      left_x = terrain_center[0] - 0.5 * terrain_size[0] + nominal_offset

      # Conventional pyramid stair geometry.
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

      tread_top_z = (k + 1) * step_height
      wall_z = tread_top_z + 0.5 * wall_height

      # Top and bottom faces climb along y. Their corridor walls are y-aligned
      # and remain at fixed x positions over every stair level.
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

      # Left and right faces climb along x. Their corridor walls are x-aligned
      # and remain at fixed y positions over every stair level.
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

    center_dims = (
      terrain_size[0] - 2.0 * num_steps * step_width,
      terrain_size[1] - 2.0 * num_steps * step_width,
      (num_steps + 2) * step_height,
    )
    center_pos = (
      terrain_center[0],
      terrain_center[1],
      terrain_center[2] + 0.5 * num_steps * step_height,
    )
    center_color = brand_ramp(_MUJOCO_PURPLE, 1.0)
    add_box(center_dims, center_pos, center_color)

    origin = np.array(
      [terrain_center[0], terrain_center[1], (num_steps + 1) * step_height]
    )
    geometries = [
      TerrainGeometry(geom=geom, color=color)
      for geom, color in zip(boxes, colors, strict=True)
    ]
    return TerrainOutput(origin=origin, geometries=geometries)
