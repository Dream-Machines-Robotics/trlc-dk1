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
    # Counts observations where the cpp camera backend gave us a kernel
    # capture timestamp (regardless of whether OBS_ALIGN was on).
    _stats_obs_aligned: int = 0
    # Counts observations whose cam_capture_ts_ns equalled the previous obs's
    # — i.e. the recording loop fired twice between two camera frames, so
    # this row's image is a duplicate of the previous row's image (rate-
    # aliasing). The ground-truth duplicate count for the dataset.
    _stats_obs_ref_unchanged: int = 0
    _stats_prev_cam_ts_ns: int | None = None

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
        # Poll the newest camera kernel capture timestamp across all bi-follower
        # cameras. We do this UNCONDITIONALLY (whenever the cpp backend is in
        # use) so we can record it in the timing side-file for diagnostics —
        # image staleness, rate-aliasing duplicates, image-vs-action skew all
        # work without needing alignment to be on. If any camera lacks the API
        # (e.g. opencv backend) or hasn't produced a frame yet, the list is
        # discarded and cam_capture_ts_ns stays None.
        cam_ts: list[int] = []
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
        cam_capture_ts_ns: int | None = max(cam_ts) if cam_ts else None

        # Alignment is a SEPARATE decision from "do we know the camera ts".
        # When _align_observations is True and a cam ts is available, use it
        # as the reference time T that motors / leader / per-cam reads will be
        # sampled at. Otherwise downstream code paths fall back to "latest
        # of everything" via the None sentinel.
        ref_ts_ns: int | None = cam_capture_ts_ns if self._align_observations else None

        # Expose both: ref_ts_ns is the *applied* reference (used by the record
        # loop to align the teleop snapshot); cam_capture_ts_ns is the
        # *observed* camera ts at this tick (used by the timing side-file
        # diagnostics regardless of whether alignment was on).
        self._last_observation_ref_ts_ns = ref_ts_ns
        self._last_camera_capture_ts_ns = cam_capture_ts_ns

        # Update stats — track whether the camera ts changed since last obs.
        # A streak of unchanged values means consecutive ticks read the same
        # camera frame → those rows are rate-aliasing duplicates (image-side).
        # This is the ground truth for the dashboard's "Duplicate Frames" card
        # and is meaningful whether alignment is on or off.
        self._stats_total_obs += 1
        if cam_capture_ts_ns is not None:
            self._stats_obs_aligned += 1
            if (
                self._stats_prev_cam_ts_ns is not None
                and cam_capture_ts_ns == self._stats_prev_cam_ts_ns
            ):
                self._stats_obs_ref_unchanged += 1
            self._stats_prev_cam_ts_ns = cam_capture_ts_ns
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "bi_follower obs: cam_ts=%s ref_ts=%s prev_cam=%s "
                "(total=%d, with_cam_ts=%d, cam_unchanged=%d)",
                cam_capture_ts_ns, ref_ts_ns, self._stats_prev_cam_ts_ns,
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
        """Reference time *used* by the most recent ``get_observation()``.

        ``None`` when alignment is disabled (``OBS_ALIGN=0``) or unavailable
        (e.g. opencv backend). Used by the recording loop to align the leader
        teleop snapshot to the same T that the rest of the observation was
        assembled from.

        See ``last_camera_capture_ts_ns`` for the camera timestamp that's
        always tracked when the cpp backend is in use, regardless of whether
        alignment was actually applied.
        """
        return getattr(self, "_last_observation_ref_ts_ns", None)

    @property
    def last_camera_capture_ts_ns(self) -> int | None:
        """Newest camera kernel capture timestamp observed at the most recent
        ``get_observation()``, regardless of the alignment toggle.

        ``None`` only if the cameras don't expose ``latest_capture_time_ns``
        (i.e. the opencv backend) or none have produced a frame yet. The
        recording loop writes this to the timing side-file so the dashboard
        can compute image-staleness and rate-aliasing duplicates without
        decoding any video, even when ``OBS_ALIGN=0``.
        """
        return getattr(self, "_last_camera_capture_ts_ns", None)

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
