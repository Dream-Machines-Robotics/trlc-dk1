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

from dataclasses import dataclass
import logging
import os
import time
import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)

logger = logging.getLogger(__name__)

# Hardware revision selector — "new" (default) matches the current shipping
# leader (mixed xl330-m288/m077 → motor models 1200/1190). "old" matches the
# earlier leader where every joint is xl330-m077 (all 1190). Set via the
# LEADER_HW_REV env var; the Makefile exports it from `make.local`.
_HW_REV_MOTORS = {
    "new": {
        "joint_1": ("xl330-m288", 1),
        "joint_2": ("xl330-m288", 2),
        "joint_3": ("xl330-m077", 3),
        "joint_4": ("xl330-m077", 4),
        "joint_5": ("xl330-m288", 5),
        "joint_6": ("xl330-m288", 6),
        "gripper": ("xl330-m077", 7),
    },
    "old": {
        "joint_1": ("xl330-m077", 1),
        "joint_2": ("xl330-m077", 2),
        "joint_3": ("xl330-m077", 3),
        "joint_4": ("xl330-m077", 4),
        "joint_5": ("xl330-m077", 5),
        "joint_6": ("xl330-m077", 6),
        "gripper": ("xl330-m077", 7),
    },
}


@TeleoperatorConfig.register_subclass("dk1_leader")
@dataclass
class DK1LeaderConfig(TeleoperatorConfig):
    port: str
    gripper_open_pos: int = 2280
    gripper_closed_pos: int = 1670
    # Bounds a single lost status packet to ~read_timeout_ms before falling through;
    # paired with read_num_retry=0 so a glitch is a sub-frame skip the C++ RT loop rides
    # out instead of a multi-frame freeze. Defaults match the ledream Makefile.
    read_timeout_ms: int = 25
    read_num_retry: int = 0


class DK1Leader(Teleoperator):
    config_class = DK1LeaderConfig
    name = "dk1_leader"

    def __init__(self, config: DK1LeaderConfig):
        super().__init__(config)
        self.config = config
        hw_rev = os.environ.get("LEADER_HW_REV", "new").strip().lower()
        if hw_rev not in _HW_REV_MOTORS:
            raise ValueError(
                f"LEADER_HW_REV must be one of {sorted(_HW_REV_MOTORS)}, got {hw_rev!r}"
            )
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                name: Motor(motor_id, model, MotorNormMode.DEGREES)
                for name, (model, motor_id) in _HW_REV_MOTORS[hw_rev].items()
            },
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = False) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        self.bus.default_timeout = self.config.read_timeout_ms
        self.bus.set_timeout(self.config.read_timeout_ms)
        self.configure()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        
        # Enable torque and set to position to open
        self.bus.write("Torque_Enable", "gripper", 0, normalize=False)
        self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value, normalize=False)
        self.bus.write("Current_Limit", "gripper", 100, normalize=False)
        self.bus.write("Torque_Enable", "gripper", 1, normalize=False)
        self.bus.write("Goal_Position", "gripper", self.config.gripper_open_pos, normalize=False)
        
    def setup_motors(self) -> None:
        for motor in self.bus.motors:
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        
        action = self.bus.sync_read(
            normalize=False, data_name="Present_Position", num_retry=self.config.read_num_retry
        )
        action = {f"{motor}.pos": (val/4096*2*np.pi-np.pi) if motor != "gripper" else val for motor, val in action.items()}
        
        # # Normalize gripper position between 1 (closed) and 0 (open)
        gripper_range = self.config.gripper_open_pos - self.config.gripper_closed_pos
        action["gripper.pos"] = 1 - (action["gripper.pos"] - self.config.gripper_closed_pos) / gripper_range
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")