"""Pyramid stairs with low longitudinal barriers.

The barriers run in the stair-climbing direction and form loose lanes on each
stair face. They restrict lateral foot escape and diagonal sliding while leaving
forward stair traversal open.
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
class BoxHurdlePyramidStairsTerrainCfg(SubTerrainCfg):
  """Pyramid stairs with low barriers aligned with the climbing direction.

  The existing ``hurdle_*`` parameter names are retained for compatibility:
  ``hurdle_depth`` is the wall thickness and ``hurdle_setback`` is the lateral
  edge margin used when laying out lane dividers.
  """

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
  """Longitudinal wall height range, interpolated by difficulty."""

  hurdle_depth: float = 0.04
  """Longitudinal wall thickness, in meters."""

  hurdle_setback: float = 0.08
  """Minimum lateral margin between the outer stair edge and a wall."""

  hurdle_probability: float = 0.5
  """Probability that a stair level receives longitudinal wall segments."""

  randomize_hurdle_height_per_step: bool = True
  """Sample the wall height independently for each stair level."""

  alternate_hurdles: bool = False
  """Place wall segments deterministically on alternating stair levels."""

  lane_width: float = 0.65
  """Target spacing between adjacent longitudinal walls, in meters."""

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
    if self.lane_width <= self.hurdle_depth:
      raise ValueError("lane_width must be larger than hurdle_depth")

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

    def divider_positions(center: float, span: float) -> list[float]:
      usable_span = span - 2.0 * self.hurdle_setback
      if usable_span <= self.lane_width:
        return []
      num_lanes = max(1, int(np.floor(usable_span / self.lane_width)))
      actual_lane_width = usable_span / num_lanes
      first_edge = center - 0.5 * usable_span
      return [first_edge + i * actual_lane_width for i in range(1, num_lanes)]

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

      # Conventional pyramid stair boxes.
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

      # Top and bottom faces climb along y, so the walls are y-aligned.
      for x_pos in divider_positions(terrain_center[0], remaining_x):
        add_box(
          (self.hurdle_depth, step_width, hurdle_height),
          (x_pos, top_y, hurdle_z),
          _BARRIER_COLOR,
        )
        add_box(
          (self.hurdle_depth, step_width, hurdle_height),
          (x_pos, bottom_y, hurdle_z),
          _BARRIER_COLOR,
        )

      # Left and right faces climb along x, so the walls are x-aligned.
      for y_pos in divider_positions(terrain_center[1], side_span_y):
        add_box(
          (step_width, self.hurdle_depth, hurdle_height),
          (right_x, y_pos, hurdle_z),
          _BARRIER_COLOR,
        )
        add_box(
          (step_width, self.hurdle_depth, hurdle_height),
          (left_x, y_pos, hurdle_z),
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
