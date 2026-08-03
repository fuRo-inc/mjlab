"""Pyramid stairs with low barriers on the treads.

The barriers are placed behind the stair nosings so that sliding a low foot over
the nosing is insufficient: the foot must retain clearance after entering the
tread. The base stair geometry remains a conventional pyramid staircase.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_border
from mjlab.utils.color import brand_ramp, darken_rgba

_MUJOCO_PURPLE = (0.58, 0.36, 0.90)
_BARRIER_COLOR = (0.90, 0.45, 0.20)
_MIN_BORDER_HEIGHT = 0.05
_MIN_GEOM_SIZE = 1.0e-6


@dataclass(kw_only=True)
class BoxHurdlePyramidStairsTerrainCfg(SubTerrainCfg):
  """Pyramid stairs with low full-width barriers on selected treads."""

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

  hurdle_height_range: tuple[float, float] = (0.03, 0.07)
  """Barrier height range above the tread, interpolated by difficulty."""

  hurdle_depth: float = 0.06
  """Barrier thickness along the stair travel direction."""

  hurdle_setback: float = 0.08
  """Distance from the stair nosing to the near face of the barrier."""

  hurdle_probability: float = 0.5
  """Probability that a given stair level receives barriers."""

  randomize_hurdle_height_per_step: bool = True
  """Sample each barrier level from zero to the difficulty-scaled height."""

  alternate_hurdles: bool = False
  """Place barriers deterministically on alternating stair levels."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    if self.border_width < 0.0:
      raise ValueError("border_width must be non-negative")
    if self.platform_width <= 0.0:
      raise ValueError("platform_width must be positive")
    if self.hurdle_height_range[0] < 0.0:
      raise ValueError("hurdle_height_range minimum must be non-negative")
    if self.hurdle_height_range[1] < self.hurdle_height_range[0]:
      raise ValueError("hurdle_height_range must satisfy min <= max")
    if self.hurdle_depth <= 0.0:
      raise ValueError("hurdle_depth must be positive")
    if self.hurdle_setback < 0.0:
      raise ValueError("hurdle_setback must be non-negative")
    if not 0.0 <= self.hurdle_probability <= 1.0:
      raise ValueError("hurdle_probability must be in [0, 1]")

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
    if self.hurdle_setback + self.hurdle_depth >= step_width:
      raise ValueError(
        "hurdle_setback + hurdle_depth must be smaller than the tread depth"
      )

    max_hurdle_height = self.hurdle_height_range[0] + difficulty * (
      self.hurdle_height_range[1] - self.hurdle_height_range[0]
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

    for k in range(num_steps):
      step_color = brand_ramp(_MUJOCO_PURPLE, k / max(num_steps - 1, 1))
      remaining_x = terrain_size[0] - 2.0 * k * step_width
      remaining_y = terrain_size[1] - 2.0 * k * step_width
      box_z = terrain_center[2] + 0.5 * k * step_height
      box_height = (k + 2) * step_height
      nominal_offset = (k + 0.5) * step_width

      # Conventional stair boxes.
      add_box(
        (remaining_x, step_width, box_height),
        (
          terrain_center[0],
          terrain_center[1] + 0.5 * terrain_size[1] - nominal_offset,
          box_z,
        ),
        step_color,
      )
      add_box(
        (remaining_x, step_width, box_height),
        (
          terrain_center[0],
          terrain_center[1] - 0.5 * terrain_size[1] + nominal_offset,
          box_z,
        ),
        step_color,
      )

      side_span_y = max(_MIN_GEOM_SIZE, remaining_y - 2.0 * step_width)
      add_box(
        (step_width, side_span_y, box_height),
        (
          terrain_center[0] + 0.5 * terrain_size[0] - nominal_offset,
          terrain_center[1],
          box_z,
        ),
        step_color,
      )
      add_box(
        (step_width, side_span_y, box_height),
        (
          terrain_center[0] - 0.5 * terrain_size[0] + nominal_offset,
          terrain_center[1],
          box_z,
        ),
        step_color,
      )

      if self.alternate_hurdles:
        place_hurdle = (k % 2) == 0
      else:
        place_hurdle = rng.random() < self.hurdle_probability
      if not place_hurdle or max_hurdle_height <= 0.0:
        continue

      hurdle_height = (
        rng.uniform(0.0, max_hurdle_height)
        if self.randomize_hurdle_height_per_step
        else max_hurdle_height
      )
      if hurdle_height <= _MIN_GEOM_SIZE:
        continue

      tread_top_z = (k + 1) * step_height
      hurdle_z = tread_top_z + 0.5 * hurdle_height
      inward = self.hurdle_setback + 0.5 * self.hurdle_depth

      # Top face: travel toward -y.
      top_outer_y = terrain_center[1] + 0.5 * terrain_size[1] - k * step_width
      add_box(
        (remaining_x, self.hurdle_depth, hurdle_height),
        (terrain_center[0], top_outer_y - inward, hurdle_z),
        _BARRIER_COLOR,
      )

      # Bottom face: travel toward +y.
      bottom_outer_y = terrain_center[1] - 0.5 * terrain_size[1] + k * step_width
      add_box(
        (remaining_x, self.hurdle_depth, hurdle_height),
        (terrain_center[0], bottom_outer_y + inward, hurdle_z),
        _BARRIER_COLOR,
      )

      # Right face: travel toward -x.
      right_outer_x = terrain_center[0] + 0.5 * terrain_size[0] - k * step_width
      add_box(
        (self.hurdle_depth, side_span_y, hurdle_height),
        (right_outer_x - inward, terrain_center[1], hurdle_z),
        _BARRIER_COLOR,
      )

      # Left face: travel toward +x.
      left_outer_x = terrain_center[0] - 0.5 * terrain_size[0] + k * step_width
      add_box(
        (self.hurdle_depth, side_span_y, hurdle_height),
        (left_outer_x + inward, terrain_center[1], hurdle_z),
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
