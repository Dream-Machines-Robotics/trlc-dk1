#   Copyright 2025 The Robot Learning Company UG (haftungsbeschränkt). All rights reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from dataclasses import dataclass, field
from functools import cached_property
import os
import time
import logging
from typing import Any

from lerobot.cameras import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig

from lerobot_robot_trlc_dk1.motors.DM_Control_Python.DM_CAN import *
from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig

logger = logging.getLogger(__name__)


def map_range(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


@RobotConfig.register_subclass("bi_dk1_follower")
@dataclass
class BiDK1FollowerConfig(RobotConfig):
    left_arm_port: str
    right_arm_port: str
    disable_torque_on_disconnect: bool = False
    joint_velocity_scaling: float = 0.2
    max_gripper_torque: float = 1.0 # Nm (/0.00875m spur gear radius = 114N gripper force)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Control mode: "pos_vel" (Python serial) or "rt_impedance" (C++ RT loop at 250Hz)
    control_mode: str = "rt_impedance"


class BiDK1Follower(Robot):
    """
    Bimanual TRLC-DK1 Follower Arm designed by The Robot Learning Company.
    """

    config_class = BiDK1FollowerConfig
    name = "bi_dk1_follower"

    def __init__(self, config: BiDK1FollowerConfig):
        super().__init__(config)

        self.config = config

        # Observation-alignment toggle. Default OFF (matches upstream LeRobot
        # "latest-of-everything-per-tick" semantics). Set LEROBOT_OBS_ALIGN=1 in
        # the environment to instead pick a reference time T (= newest camera
        # kernel capture timestamp) and source motor + leader state from T, so
        # every dataset row contains a self-consistent (image, motors, action)
        # triple at the cost of one camera period of inherent latency.
        _align_env = os.environ.get("LEROBOT_OBS_ALIGN", "0").strip().lower()
        self._align_observations = _align_env in ("1", "true", "yes", "on")

        left_arm_config = DK1FollowerConfig(
            port=self.config.left_arm_port,
            disable_torque_on_disconnect=self.config.disable_torque_on_disconnect,
            joint_velocity_scaling=self.config.joint_velocity_scaling,
            max_gripper_torque=self.config.max_gripper_torque,
            control_mode=self.config.control_mode,
        )
        right_arm_config = DK1FollowerConfig(
            port=self.config.right_arm_port,
            disable_torque_on_disconnect=self.config.disable_torque_on_disconnect,
            joint_velocity_scaling=self.config.joint_velocity_scaling,
            max_gripper_torque=self.config.max_gripper_torque,
            control_mode=self.config.control_mode,
        )
        
        self.left_arm = DK1Follower(left_arm_config)
        self.right_arm = DK1Follower(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.motors
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self) -> None:
        self.left_arm.connect()
        self.right_arm.connect()

        try:
            for cam in self.cameras.values():
                cam.connect()
        except Exception:
            # Clean up RT loops if camera connection fails
            self.left_arm.disconnect()
            self.right_arm.disconnect()
            raise

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def get_joint_torques(self) -> dict[str, float]:
        """Per-arm torque side-channel for haptic feedback. Mirrors
        ``DK1Follower.get_joint_torques`` with ``left_/right_`` prefixes."""
        out: dict[str, float] = {}
        for key, val in self.left_arm.get_joint_torques().items():
            out[f"left_{key}"] = val
        for key, val in self.right_arm.get_joint_torques().items():
            out[f"right_{key}"] = val
        return out

    # Stats for diagnostics (read by lerobot_record to surface in status line
    # or end-of-episode summary). All counters are reset on connect().
    _stats_total_obs: int = 0
    _stats_obs_aligned: int = 0          # ref_ts_ns was determinable
    _stats_obs_ref_unchanged: int = 0    # same ref_ts_ns as previous obs (= duplicate)
    _stats_prev_ref_ts_ns: int | None = None

    def get_observation_stats(self) -> dict[str, int | float]:
        """Return cumulative observation-assembly stats since connect().

        Useful for detecting frame-rate aliasing duplicates: when
        ``ref_unchanged`` / ``total`` is non-trivial, the record loop is
        firing faster than the cameras produce new frames, so consecutive
        ticks read the same reference time → same image, motors, action →
        a duplicate row.
        """
        return {
            "total": self._stats_total_obs,
            "aligned": self._stats_obs_aligned,
            "ref_unchanged": self._stats_obs_ref_unchanged,
            "ref_unchanged_pct": (
                100.0 * self._stats_obs_ref_unchanged / self._stats_total_obs
                if self._stats_total_obs > 0
                else 0.0
            ),
        }

    def get_observation(self) -> dict[str, Any]:
        """Return a single dict of all sensors, optionally time-aligned.

        When ``self._align_observations`` is True *and* every bi-follower
        camera exposes ``latest_capture_time_ns`` (true for the cpp backend,
        false for opencv), we pick a reference time ``T`` = the newest camera
        frame's kernel capture timestamp and ask every other sensor — motors
        via the RT loop's state ring, per-arm cameras — for state at-or-before
        ``T``. The dataset row then contains a self-consistent snapshot from
        ~1 camera period in the past, instead of "latest of everything at tick
        time" (which can drift apart during Python catch-up).

        The chosen ``T`` is stashed on ``self._last_observation_ref_ts_ns`` so
        the recording loop can use it to align the leader teleop snapshot too.

        When alignment is disabled, or when a camera doesn't carry kernel
        timestamps, falls back to legacy latest semantics on every read.
        """
        # Pick the reference time T — newest camera kernel capture timestamp
        # across all bi-follower cameras. If any camera lacks the API or has
        # never produced a frame, fall back to latest-of-everything.
        ref_ts_ns: int | None = None
        cam_ts: list[int] = []
        if self._align_observations:
            for cam in self.cameras.values():
                get_ts = getattr(cam, "latest_capture_time_ns", None)
                if get_ts is None:
                    cam_ts = []
                    break
                ts = get_ts()
                if ts is None:
                    cam_ts = []
                    break
                cam_ts.append(ts)
            if cam_ts:
                ref_ts_ns = max(cam_ts)
        # If _align_observations is False, ref_ts_ns stays None — every
        # downstream call (per-arm get_observation, cam.read_at_or_before)
        # already handles None as "use legacy latest semantics".

        # Expose to callers (e.g. the recording loop) so they can align the
        # leader teleop snapshot to the same T.
        self._last_observation_ref_ts_ns = ref_ts_ns

        # Update stats — track whether ref_ts_ns changed since last obs. A
        # streak of unchanged refs means consecutive ticks read the same
        # frame/motors/action triple → those rows are duplicates in the
        # dataset, almost always caused by 50 Hz loop vs 50 fps camera
        # rate aliasing (not by a logic bug).
        self._stats_total_obs += 1
        if ref_ts_ns is not None:
            self._stats_obs_aligned += 1
            if (
                self._stats_prev_ref_ts_ns is not None
                and ref_ts_ns == self._stats_prev_ref_ts_ns
            ):
                self._stats_obs_ref_unchanged += 1
            self._stats_prev_ref_ts_ns = ref_ts_ns
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "bi_follower obs: ref_ts_ns=%s cams=%s prev=%s "
                "(total=%d, aligned=%d, ref_unchanged=%d)",
                ref_ts_ns, cam_ts, self._stats_prev_ref_ts_ns,
                self._stats_total_obs, self._stats_obs_aligned,
                self._stats_obs_ref_unchanged,
            )

        obs_dict: dict[str, Any] = {}

        left_obs = self.left_arm.get_observation(ref_ts_ns=ref_ts_ns)
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        right_obs = self.right_arm.get_observation(ref_ts_ns=ref_ts_ns)
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            frame = None
            if ref_ts_ns is not None and hasattr(cam, "read_at_or_before"):
                frame = cam.read_at_or_before(ref_ts_ns)
            if frame is None:
                # Fallback: legacy non-blocking latest read.
                frame = cam.read_latest()
            obs_dict[cam_key] = frame
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @property
    def last_observation_ref_ts_ns(self) -> int | None:
        """Reference time used by the most recent ``get_observation()``.

        ``None`` if no observation has been taken yet, or if the cameras
        don't support timestamp alignment (legacy / opencv backend). Used
        by the recording loop to align the leader teleop snapshot to the
        same T the rest of the observation was assembled from.
        """
        return getattr(self, "_last_observation_ref_ts_ns", None)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        prefixed_send_action_left = {f"left_{key}": value for key, value in send_action_left.items()}
        prefixed_send_action_right = {f"right_{key}": value for key, value in send_action_right.items()}

        return {**prefixed_send_action_left, **prefixed_send_action_right}

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()
