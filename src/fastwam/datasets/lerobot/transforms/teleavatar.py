"""Teleavatar / TA2 action-state selection (aligned with openpi teleavatar_v2).

Raw dataset layout (72-d shared by action & state):
  positions[0:16]  = [L_arm(7), L_grip_pos, R_arm(7), R_grip_pos]
  velocities[16:32]
  efforts[32:48]   = [L_arm(7), L_grip_effort, R_arm(7), R_grip_effort]
  EE[48:62]        = [L_pos(3), L_quat(4), R_pos(3), R_quat(4)]
  chassis[62:71]   = [pos(3), vel(3), eff(3)]
  kinco[71]        = lifting-column velocity -- command in `action`,
                     measured velocity in `observation.state`

Model formats (openpi convention):
  state  14-d = [L_arm(7), R_arm(7)]
  action 16-d = [L_arm(7), L_grip, R_arm(7), R_grip]

The kinco channel is appended independently per field, so three combinations
are reachable (`lift_in_action` / `lift_in_state`):

  off / off  ->  action 16-d, state 14-d          `lift=off`
  on  / off  ->  action 17-d, state 14-d          `lift=on`   (recommended)
  on  / on   ->  action 17-d, state 15-d          legacy, see `include_lift`

  action 17-d = [L_arm(7), L_grip, R_arm(7), R_grip, lift_vel_cmd]
  state  15-d = [L_arm(7), R_arm(7), lift_vel_measured]

Why `lift=on` leaves lift out of *state*
----------------------------------------
Developer documentation §4.8 lists every public state topic -- arm, gripper
and chassis `joint_states` -- and there is no lifting-column state topic. The
measured velocity therefore cannot be read back on a real robot: the client
pins raw `state[71]` to 0 (see `arm_config.yml` `lift.state_topic`). Training
on the live signal is actively harmful, because the measured velocity tracks
the command almost perfectly (lag-0 correlation 0.9978 in our lift data), so the
policy learns to copy that channel instead of reading the column off the
images. At deploy time the constant 0 then reads as "column is stationary" and
the policy keeps commanding zero. Keeping lift out of the state removes the
shortcut and the train/deploy mismatch in one move.

`include_lift=True` is kept as a deprecated alias for the legacy on/on pair so
runs recorded before the split (2026-09-12) stay reproducible.

Gripper channels in action are force-control: dataset stores effort (Nm);
before normalization we convert effort -> platform trigger in [0, 1];
backward converts trigger -> effort for robot-side publishing.

The lift channel is passed through in raw dataset units (no rescale) -- the
min/max normalizer already maps it to [-1, 1]. Note the recorded command
saturates at +/-150, while `/api/servo/cmd` takes a normalized velocity in
[-1, 1]; robot-side publishing therefore has to divide by that saturation
value. Datasets that never moved the column (e.g. the MK_tower floors) store
an all-zero kinco channel, which the normalizer's degenerate-range guard maps
to a constant 0 rather than dividing by zero.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


def gripper_effort_to_trigger(effort: torch.Tensor) -> torch.Tensor:
    """Dataset gripper effort (Nm) -> platform trigger [≈0, 1] (TA2 / openpi v2)."""
    return torch.where(
        effort > 0,
        0.10 * (1.0 - effort / 2.0),
        0.10 - effort * 0.90 / 1.6,
    )


def gripper_trigger_to_effort(trigger: torch.Tensor) -> torch.Tensor:
    """Platform trigger [0, 1] -> gripper effort (Nm) (TA2 / openpi v2)."""
    return torch.where(
        trigger < 0.10,
        2.0 * (1.0 - trigger / 0.10),
        -1.6 * (trigger - 0.10) / 0.90,
    )


class TeleavatarSelectTransform:
    """Slice 72-d Teleavatar vectors to 14/15-d state and 16/17-d action.

    Args:
        keys: shape_meta keys to transform (usually ["default"]).
        convert_gripper_to_trigger: if True, map action gripper effort->trigger
            on forward (before norm) and trigger->effort on backward.
        lift_in_action: append the kinco lifting-column command to the action,
            giving 17-d. Only turn this on for datasets that actually drive the
            column -- see the module docstring.
        lift_in_state: append the kinco measured velocity to the state, giving
            15-d. Defaults to False even when `lift_in_action` is True, because
            no public topic can supply it at deploy time (§4.8) -- see the
            module docstring before switching it on.
        include_lift: DEPRECATED alias that sets both flags at once. Kept so
            pre-split runs stay reproducible; prefer the `lift=on|off` config
            group for new runs.
    """

    # Indices into the raw 72-d vector.
    STATE_IDX = list(range(0, 7)) + list(range(8, 15))  # 14
    ACTION_ARM_L = list(range(0, 7))
    ACTION_ARM_R = list(range(8, 15))
    ACTION_GRIP_L = 39
    ACTION_GRIP_R = 47
    LIFT_IDX = 71  # kinco_velocity

    def __init__(
        self,
        keys: Optional[List[str]] = None,
        convert_gripper_to_trigger: bool = True,
        lift_in_action: bool = False,
        lift_in_state: bool = False,
        include_lift: Optional[bool] = None,
    ):
        self.keys = keys if keys is not None else ["default"]
        self.convert_gripper_to_trigger = bool(convert_gripper_to_trigger)

        if include_lift is not None:
            # Pre-split configs drove both fields off one flag. Honour it rather than
            # silently dropping the lift channel, but say so once -- a 15-d state is
            # not servable (§4.8) and the caller probably wants `lift=on` instead.
            if lift_in_action or lift_in_state:
                raise ValueError(
                    "`include_lift` is a deprecated alias for `lift_in_action` + "
                    "`lift_in_state`; pass either the alias or the two flags, not both."
                )
            lift_in_action = lift_in_state = bool(include_lift)
            if include_lift:
                logger.warning(
                    "TeleavatarSelectTransform(include_lift=True) is deprecated: it puts the "
                    "lifting-column *measured velocity* into the state, which no public topic "
                    "can supply at deploy time (§4.8), so state[71] is pinned to 0 on the robot. "
                    "Use the `lift=on` config group (action 17-d / state 14-d) unless you are "
                    "deliberately reproducing a pre-split run."
                )

        self.lift_in_action = bool(lift_in_action)
        self.lift_in_state = bool(lift_in_state)
        # Retained for configs/tools that still read it; True only for the legacy pair.
        self.include_lift = self.lift_in_action and self.lift_in_state

        self.state_idx = list(self.STATE_IDX)
        if self.lift_in_state:
            self.state_idx.append(self.LIFT_IDX)
        # 7 + 1 + 7 + 1 (+ 1 lift)
        self.action_dim = 17 if self.lift_in_action else 16
        self.state_dim = len(self.state_idx)
        # Raw width forward() has to be able to index into, per field. The two fields are
        # sliced independently, so each gets its own bound -- a 17-d action with a 14-d
        # state must not demand 72 raw columns from the state vector.
        lift_min = self.LIFT_IDX + 1
        self.min_raw_action_dim = lift_min if self.lift_in_action else self.ACTION_GRIP_R + 1
        self.min_raw_state_dim = lift_min if self.lift_in_state else max(self.STATE_IDX) + 1

    def forward(self, batch: Dict) -> Dict:
        for k in self.keys:
            if "action" in batch and k in batch["action"]:
                a = batch["action"][k]
                assert a.shape[-1] >= self.min_raw_action_dim, f"action '{k}' too short: {a.shape}"
                parts = [
                    a[..., self.ACTION_ARM_L],
                    a[..., self.ACTION_GRIP_L : self.ACTION_GRIP_L + 1],
                    a[..., self.ACTION_ARM_R],
                    a[..., self.ACTION_GRIP_R : self.ACTION_GRIP_R + 1],
                ]
                if self.lift_in_action:
                    parts.append(a[..., self.LIFT_IDX : self.LIFT_IDX + 1])
                selected = torch.cat(parts, dim=-1)
                if self.convert_gripper_to_trigger:
                    selected = selected.clone()
                    selected[..., 7] = gripper_effort_to_trigger(selected[..., 7])
                    selected[..., 15] = gripper_effort_to_trigger(selected[..., 15])
                batch["action"][k] = selected

            if "state" in batch and k in batch["state"]:
                s = batch["state"][k]
                assert s.shape[-1] >= self.min_raw_state_dim, f"state '{k}' too short: {s.shape}"
                idx = torch.as_tensor(self.state_idx, device=s.device, dtype=torch.long)
                batch["state"][k] = s.index_select(dim=-1, index=idx)

        return batch

    def backward(self, batch: Dict) -> Dict:
        """Convert action gripper trigger -> effort; keep the selected width (no expand to 72)."""
        if "action" not in batch or not self.convert_gripper_to_trigger:
            return batch
        for k in self.keys:
            if k not in batch["action"]:
                continue
            a = batch["action"][k]
            assert a.shape[-1] == self.action_dim, (
                f"expected {self.action_dim}-d action for backward, got {a.shape}"
            )
            a = a.clone()
            a[..., 7] = gripper_trigger_to_effort(a[..., 7])
            a[..., 15] = gripper_trigger_to_effort(a[..., 15])
            batch["action"][k] = a
        return batch
