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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import time
from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig

from lerobot_robot_trlc_dk1.leader import DK1Leader, DK1LeaderConfig

logger = logging.getLogger(__name__)


@TeleoperatorConfig.register_subclass("bi_dk1_leader")
@dataclass
class BiDK1LeaderConfig(TeleoperatorConfig):
    left_arm_port: str
    right_arm_port: str
    gripper_open_pos: int = 2280
    gripper_closed_pos: int = 1670
    # See DK1LeaderConfig — forwarded to both arms.
    read_timeout_ms: int = 25
    read_num_retry: int = 0


class BiDK1Leader(Teleoperator):
    config_class = BiDK1LeaderConfig
    name = "bi_dk1_leader"

    def __init__(self, config: BiDK1LeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = DK1LeaderConfig(
            port=self.config.left_arm_port,
            gripper_open_pos=self.config.gripper_open_pos,
            gripper_closed_pos=self.config.gripper_closed_pos,
            read_timeout_ms=self.config.read_timeout_ms,
            read_num_retry=self.config.read_num_retry,
        )
        right_arm_config = DK1LeaderConfig(
            port=self.config.right_arm_port,
            gripper_open_pos=self.config.gripper_open_pos,
            gripper_closed_pos=self.config.gripper_closed_pos,
            read_timeout_ms=self.config.read_timeout_ms,
            read_num_retry=self.config.read_num_retry,
        )

        self.left_arm = DK1Leader(left_arm_config)
        self.right_arm = DK1Leader(right_arm_config)

        # ② Read the two arms concurrently (each blocks on its own serial port,
        # releasing the GIL during the read) so a stall on one arm doesn't delay
        # the other. Sequential reads meant a glitch on either arm froze BOTH.
        # 2 persistent workers — created once, reused every 250 Hz tick.
        self._read_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bi_leader_read")
        # Last good per-arm action, so a single-arm read failure falls back to
        # that arm's last pose (the C++ RT loop holds it) instead of freezing
        # or crashing the whole teleop loop. None until the first good read.
        self._last_left: dict[str, float] | None = None
        self._last_right: dict[str, float] | None = None
        # perf_counter() of each arm's last *fresh* read — feeds the "ms since
        # last good read" diagnostic when we fall back to a held pose.
        self._last_left_t: float = 0.0
        self._last_right_t: float = 0.0

    @property
    def action_features(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = False) -> None:
        self.left_arm.connect()
        self.right_arm.connect()    

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()
        
    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_action(self) -> dict[str, float]:
        # Kick off both arm reads concurrently (each blocks on its own port).
        fut_left = self._read_pool.submit(self.left_arm.get_action)
        fut_right = self._read_pool.submit(self.right_arm.get_action)
        left, left_err = self._await(fut_left)
        right, right_err = self._await(fut_right)

        # Both arms failing in the same tick is a real fault (e.g. disconnect),
        # not a transient single-arm glitch — propagate it so the teleop
        # thread's consecutive-error safety stop still trips.
        if left_err and right_err:
            raise left_err

        # A single-arm glitch: hold that arm's last pose (the C++ RT loop keeps
        # it steady) so the healthy arm stays live. Re-raise if that arm has
        # never produced a reading — there's nothing to hold yet. The "ms since
        # last good read" tells a one-off blip (~one tick) from a sustained
        # stall on that specific arm/cable.
        now = time.perf_counter()
        if left_err:
            if self._last_left is None:
                raise left_err
            logger.warning("left leader read failed (%s) — holding last pose (%.0f ms since last good left read)",
                           left_err, (now - self._last_left_t) * 1e3)
            left = self._last_left
        else:
            self._last_left, self._last_left_t = left, now
        if right_err:
            if self._last_right is None:
                raise right_err
            logger.warning("right leader read failed (%s) — holding last pose (%.0f ms since last good right read)",
                           right_err, (now - self._last_right_t) * 1e3)
            right = self._last_right
        else:
            self._last_right, self._last_right_t = right, now

        action_dict = {f"left_{k}": v for k, v in left.items()}
        action_dict.update({f"right_{k}": v for k, v in right.items()})
        return action_dict

    @staticmethod
    def _await(future):
        """Return ``(result, None)`` on success or ``(None, exception)`` on a
        failed read — so the caller can decide single- vs double-arm policy."""
        try:
            return future.result(), None
        except Exception as e:  # noqa: BLE001 — surfaced/handled by caller
            return None, e

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        self._read_pool.shutdown(wait=False, cancel_futures=True)
        self.left_arm.disconnect()
        self.right_arm.disconnect()