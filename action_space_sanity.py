#!/usr/bin/env python3
from __future__ import annotations
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
env = (HERE / "payload/envs/isaaclab_anymal_room.py").read_text()
cfg = (HERE / "payload/configs/env/isaaclab_anymal_room.yaml").read_text()

assert "action_count: 11" in cfg
assert "lateral_speed: 0.25" in cfg
required = [
    "[forward_speed, 0.00, 0.00]",
    "[-backward_speed, 0.00, 0.00]",
    "[arc_forward_speed, 0.00, arc_yaw_rate]",
    "[arc_forward_speed, 0.00, -arc_yaw_rate]",
    "[0.00, 0.00, coarse_yaw_rate]",
    "[0.00, 0.00, -coarse_yaw_rate]",
    "[0.00, 0.00, 0.00]",
    "[0.00, 0.00, fine_yaw_rate]",
    "[0.00, 0.00, -fine_yaw_rate]",
    "[0.00, lateral_speed, 0.00]",
    "[0.00, -lateral_speed, 0.00]",
]
pos = [env.index(x) for x in required]
assert pos == sorted(pos)
assert '"left_arc", "right_arc"' in env
assert '"strafe_left", "strafe_right"' in env
assert "num_classes=self._action_count" in env
print("[ACTION SPACE SANITY] original 9 actions preserved; left/right strafes appended as 9/10")
