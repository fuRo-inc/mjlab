"""Split-offset pyramid stairs terrain.

Each stair face is divided into two lateral halves whose nosing positions are
shifted in opposite directions. The mean nosing position remains unchanged,
while a fixed diagonal traversal strategy becomes less effective.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_border
from mjlab.utils.color import brand_ramp, darken_rgba

_MUJOCO_PURPLE = (0.58, 0.36, 0.90)
_MIN_BORDER_HEIGHT = 0.05
_MIN_GEOM_SIZE = 1.0e-6


@dataclass(kw_only=True)
class BoxSplitOffsetPyramidStairsTerrainCfg(SubTerrainCfg):
  """Pyramid stairs with laterally split, oppositely offset stair nosings.

  For the top and bottom stair faces, each step is split along x. For the left
  and right faces, each step is split along y. The two halves receive offsets
  ``-d`` and ``+d`` along the stair travel direction, preserving the average
  step position.
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

  offset_range: tuple[float, float] = (0.0, 0.04)
  """Min and max half-offset magnitude, interpolated by difficulty."""

  randomize_offset_per_step: bool = True
  """Sample offset magnitude per step instead of once per sub-terrain."""

  alternate_offset_sign: bool = True
  """Reverse which lateral half leads on each consecutive step."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    if self.border_width < 0.0:
      raise ValueError("border_width must be non-negative")
    if self.platform_width <= 0.0:
      raise ValueError("platform_width must be positive")
    if self.offset_range[0] < 0.0 or self.offset_range[1] < self.offset_range[0]:
      raise ValueError("offset_range must satisfy 0 <= min <= max")

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

    max_offset = self.offset_range[0] + difficulty * (
      self.offset_range[1] - self.offset_range[0]
    )
    # Keep each half-step positive and prevent excessive overlap with neighbors.
    max_offset = min(max_offset, 0.45 * step_width)

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

    shared_offset = rng.uniform(0.0, max_offset) if max_offset > 0.0 else 0.0

    def add_box(size_xyz: tuple[float, float, float], pos_xyz: tuple[float, float, float], color):
      safe_size = tuple(max(_MIN_GEOM_SIZE, 0.5 * dim) for dim in size_xyz)
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=safe_size,
        pos=pos_xyz,
      )
      boxes.append(geom)
      colors.append(color)

    for k in range(num_steps):
      color = brand_ramp(_MUJOCO_PURPLE, k / max(num_steps - 1, 1))
      offset_mag = (
        rng.uniform(0.0, max_offset)
        if self.randomize_offset_per_step and max_offset > 0.0
        else shared_offset
      )
      sign = -1.0 if self.alternate_offset_sign and (k % 2 == 1) else 1.0
      first_offset = sign * offset_mag
      second_offset = -first_offset

      remaining_x = terrain_size[0] - 2.0 * k * step_width
      remaining_y = terrain_size[1] - 2.0 * k * step_width
      box_z = terrain_center[2] + 0.5 * k * step_height
      box_height = (k + 2) * step_height
      nominal_offset = (k + 0.5) * step_width

      # Top and bottom faces: split laterally along x and offset along y.
      half_x = 0.5 * remaining_x
      quarter_x = 0.25 * remaining_x
      for x_center, local_offset in (
        (terrain_center[0] - quarter_x, first_offset),
        (terrain_center[0] + quarter_x, second_offset),
      ):
        add_box(
          (half_x, step_width, box_height),
          (
            x_center,
            terrain_center[1] + 0.5 * terrain_size[1] - nominal_offset + local_offset,
            box_z,
          ),
          color,
        )
        add_box(
          (half_x, step_width, box_height),
          (
            x_center,
            terrain_center[1] - 0.5 * terrain_size[1] + nominal_offset - local_offset,
            box_z,
          ),
          color,
        )

      # Left and right faces: split laterally along y and offset along x.
      side_span_y = max(_MIN_GEOM_SIZE, remaining_y - 2.0 * step_width)
      half_y = 0.5 * side_span_y
      quarter_y = 0.25 * side_span_y
      for y_center, local_offset in (
        (terrain_center[1] - quarter_y, first_offset),
        (terrain_center[1] + quarter_y, second_offset),
      ):
        add_box(
          (step_width, half_y, box_height),
          (
            terrain_center[0] + 0.5 * terrain_size[0] - nominal_offset + local_offset,
            y_center,
            box_z,
          ),
          color,
        )
        add_box(
          (step_width, half_y, box_height),
          (
            terrain_center[0] - 0.5 * terrain_size[0] + nominal_offset - local_offset,
            y_center,
            box_z,
          ),
          color,
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
