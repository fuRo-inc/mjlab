.. _motion-imitation:

Motion Imitation
================

mjlab can train humanoid policies to imitate reference motions. This page
covers motion data preprocessing and training.

Motion preprocessing
--------------------

Reference motions are retargeted CSV files in Unitree's generalized
coordinate convention (base position, base quaternion in xyzw, then joint
angles).

Convert a CSV to the NPZ format mjlab expects:

.. code-block:: bash

   MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
       --input-file <PATH_TO_CSV> \
       --output-name <MOTION_NAME> \
       --input-fps 30 \
       --output-fps 50 \
       --render True

The script plays the motion through MuJoCo Warp, computes forward kinematics
for every body, and saves the resulting NPZ locally at ``--output-name``. With rendering
enabled, the MP4 is saved beside it.

.. warning::

   You **must** use mjlab's converter (``mjlab.scripts.csv_to_npz``).
   Converters from other frameworks such as IsaacLab produce NPZ files with
   incompatible body orderings. The NPZ stores precomputed body positions and
   quaternions indexed by body number, and different physics engines assign
   body indices differently (MuJoCo uses depth first traversal, PhysX uses
   breadth first). A mismatched NPZ will map tracking targets to the wrong
   bodies and training will not converge.

Training
--------

.. code-block:: bash

   uv run train Mjlab-Tracking-Flat-Unitree-G1 \
       --motion-file /path/to/motion.npz \
       --env.scene.num-envs 4096

Evaluation
----------

.. code-block:: bash

   uv run play Mjlab-Tracking-Flat-Unitree-G1 \
       --checkpoint-file logs/rsl_rl/experiment/run/model_1000.pt
