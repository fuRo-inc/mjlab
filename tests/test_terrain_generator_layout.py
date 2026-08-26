from __future__ import annotations

import numpy as np

from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg


def test_curriculum_layout_axes_and_origin_indexing() -> None:
  cfg = TerrainGeneratorCfg(
    curriculum=True,
    size=(4.0, 6.0),
    num_rows=3,
    num_cols=2,
    sub_terrains={
      "flat_a": BoxFlatTerrainCfg(proportion=0.5),
      "flat_b": BoxFlatTerrainCfg(proportion=0.5),
    },
  )
  generator = TerrainGenerator(cfg)

  # The public lookup stays [level, terrain_type]. World placement is checked
  # at the sub-terrain corner because individual terrain origins may be shifted
  # internally by the terrain implementation.
  assert generator.terrain_origins.shape == (3, 2, 3)
  np.testing.assert_allclose(
    generator._get_sub_terrain_position(0, 1)
    - generator._get_sub_terrain_position(0, 0),
    np.array([4.0, 0.0, 0.0]),
  )
  np.testing.assert_allclose(
    generator._get_sub_terrain_position(1, 0)
    - generator._get_sub_terrain_position(0, 0),
    np.array([0.0, 6.0, 0.0]),
  )
