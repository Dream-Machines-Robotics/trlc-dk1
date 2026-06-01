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

        # Two persistent workers (created once, reused every 250 Hz tick) so the
        # two arms' serial reads run concurrently instead of sequentially: each
        # read blocks on its own port and releases the GIL, so one arm's read
        # latency (e.g. waiting out read_timeout_ms on a lost status packet)
        # doesn't delay the other.
        self._read_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bi_leader_read")

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
        # Submit both reads before awaiting either, so the two arms' serial
        # reads run concurrently. A failed read on either arm propagates to the
        # caller: the teleop loop's consecutive-error safety stop handles it,
        # with the C++ RT loop holding the last target meanwhile — the same
        # failure policy as a sequential read, minus the cross-arm latency.
        fut_left = self._read_pool.submit(self.left_arm.get_action)
        fut_right = self._read_pool.submit(self.right_arm.get_action)
        left = fut_left.result()
        right = fut_right.result()

        action_dict = {f"left_{k}": v for k, v in left.items()}
        action_dict.update({f"right_{k}": v for k, v in right.items()})
        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        self._read_pool.shutdown(wait=False, cancel_futures=True)
        self.left_arm.disconnect()
        self.right_arm.disconnect()
