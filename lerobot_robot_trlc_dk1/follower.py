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
import serial
import time
import logging
from typing import Any

from lerobot.cameras import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot_robot_trlc_dk1.motors.DM_Control_Python.DM_CAN import *

logger = logging.getLogger(__name__)


def map_range(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


@RobotConfig.register_subclass("dk1_follower")
@dataclass
class DK1FollowerConfig(RobotConfig):
    port: str
    disable_torque_on_disconnect: bool = False
    joint_velocity_scaling: float = 0.2
    max_gripper_torque: float = 1.0 # Nm (/0.00875m spur gear radius = 114N gripper force)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Control mode: "pos_vel" (Python serial) or "rt_impedance" (C++ RT loop at 250Hz)
    control_mode: str = "rt_impedance"


class DK1Follower(Robot):
    """
    TRLC-DK1 Follower Arm designed by The Robot Learning Company.
    """

    config_class = DK1FollowerConfig
    name = "dk1_follower"

    def __init__(self, config: DK1FollowerConfig):
        super().__init__(config)
        
        # Constants for EMIT control
        self.DM4310_TORQUE_CONSTANT = 0.945  # Nm/A
        self.EMIT_VELOCITY_SCALE = 100  # rad/s
        self.EMIT_CURRENT_SCALE = 1000  # A
        
        self.JOINT_LIMITS = {
            "joint_4": (-100/180*np.pi, 100/180*np.pi),
            "joint_5": (-90/180*np.pi, 90/180*np.pi),
        }
        
        self.DM4310_SPEED = 200/60*2*np.pi   # rad/s (200  rpm | 20.94 rad/s)
        self.DM4340_SPEED = 52.5/60*2*np.pi  # rad/s (52.5 rpm | 5.49  rad/s)

        self.config = config
        self.motors = {
            "joint_1": Motor(DM_Motor_Type.DM4340, 0x01, 0x11),
            "joint_2": Motor(DM_Motor_Type.DM4340, 0x02, 0x12),
            "joint_3": Motor(DM_Motor_Type.DM4340, 0x03, 0x13),
            "joint_4": Motor(DM_Motor_Type.DM4310, 0x04, 0x14),
            "joint_5": Motor(DM_Motor_Type.DM4310, 0x05, 0x15),
            "joint_6": Motor(DM_Motor_Type.DM4310, 0x06, 0x16),
            "gripper": Motor(DM_Motor_Type.DM4310, 0x07, 0x17),
        }
        self.control = None
        self.serial_device = None
        self.bus_connected = False
        self._rt_robot = None  # DK1RobotRT instance when using rt_impedance mode

        self.gripper_open_pos = 0.0
        self.gripper_closed_pos = -4.7

        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

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
        bus_ok = self.bus_connected or (self._rt_robot is not None)
        return bus_ok and all(cam.is_connected for cam in self.cameras.values())

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if self.config.control_mode == "rt_impedance":
            self._connect_rt()
        else:
            self.serial_device = serial.Serial(
                self.config.port, 921600, timeout=0.5)
            time.sleep(0.5)

            self.control = MotorControl(self.serial_device)
            self.bus_connected = True
            self.configure()

        for cam in self.cameras.values():
            cam.connect()

    def _connect_rt(self) -> None:
        """Connect using the C++ RT control loop."""
        from trlc_dk1_control.config import DK1RobotConfig
        from trlc_dk1_control.rt_robot import DK1RobotRT

        rt_config = DK1RobotConfig(
            serial_port=self.config.port,
            disable_torque_on_disconnect=self.config.disable_torque_on_disconnect,
            max_gripper_torque_nm=self.config.max_gripper_torque,
            joint_velocity_scaling=self.config.joint_velocity_scaling,
        )
        self._rt_robot = DK1RobotRT(rt_config)
        self._rt_robot.connect()
        self.bus_connected = True
        logger.info(
            f"{self} connected via C++ RT control loop "
            f"(joint_velocity_scaling={self.config.joint_velocity_scaling})"
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:

        for key, motor in self.motors.items():
            self.control.addMotor(motor)

            for _ in range(3):
                self.control.refresh_motor_status(motor)
                time.sleep(0.01)

            if self.control.read_motor_param(motor, DM_variable.CTRL_MODE) is not None:
                logger.info(f"{key} ({motor.MotorType.name}) is connected.")

                self.control.switchControlMode(motor, Control_Type.POS_VEL)
                self.control.enable(motor)
            else:
                raise Exception(
                    f"Unable to read from {key} ({motor.MotorType.name}).")

        for joint in ["joint_1", "joint_2", "joint_3"]:
            self.control.change_motor_param(self.motors[joint], DM_variable.ACC, 10.0)
            self.control.change_motor_param(self.motors[joint], DM_variable.DEC, -10.0)
            self.control.change_motor_param(self.motors[joint], DM_variable.KP_APR, 200)
            self.control.change_motor_param(self.motors[joint], DM_variable.KI_APR, 10)

        for joint in ["gripper"]:
            self.control.change_motor_param(
                self.motors[joint], DM_variable.KP_APR, 100)

        # Open gripper and set zero position
        self.control.switchControlMode(
            self.motors["gripper"], Control_Type.VEL)
        self.control.control_Vel(self.motors["gripper"], 10.0)
        while True:
            self.control.refresh_motor_status(self.motors["gripper"])
            tau = self.motors["gripper"].getTorque()
            if tau > 1.2:
                self.control.control_Vel(self.motors["gripper"], 0.0)
                self.control.disable(self.motors["gripper"])
                self.control.set_zero_position(self.motors["gripper"])
                time.sleep(0.2)
                self.control.enable(self.motors["gripper"])
                break
            time.sleep(0.01)
        self.control.switchControlMode(
            self.motors["gripper"], Control_Type.Torque_Pos)

    def get_joint_torques(self) -> dict[str, float]:
        """Per-motor torque readings (Nm), straight from the RT loop.

        NOT in ``observation_features`` on purpose: this is a side-channel
        for runtime UX (haptic feedback), not for dataset recording — adding
        torque keys to the schema would break policies trained without
        them. Returns ``{}`` outside RT mode (the Python serial path
        doesn't surface torque cheaply).

        v1 returns only the gripper torque. Arm joint torques are also
        available from ``_rt_robot.get_joint_state()["torque"]`` but are
        gravity-comp dominated at rest, which makes them unsuitable for
        "is something pushing on the arm" UX without subtracting the
        gravity model.
        """
        if self._rt_robot is None:
            return {}
        gripper_state = self._rt_robot.get_gripper_state()
        # `pos` (normalized 0..1, 0=open / 1=closed) is included alongside
        # the torque so downstream consumers can compute a numerical
        # velocity and subtract the inertia/kinetic-friction component
        # from the haptic intensity. Cheap to ship; ignored if not used.
        return {
            "gripper.torque": float(gripper_state["torque"]),
            "gripper.pos":    float(gripper_state["pos"]),
        }

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        obs_dict = {}

        if self._rt_robot is not None:
            # RT mode: read via seqlock (~1 microsecond, thread-safe)
            joint_state = self._rt_robot.get_joint_state()
            gripper_state = self._rt_robot.get_gripper_state()
            motor_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
            for i, name in enumerate(motor_names):
                obs_dict[f"{name}.pos"] = float(joint_state["pos"][i])
            obs_dict["gripper.pos"] = float(gripper_state["pos"])
        else:
            # Python serial mode
            for key, motor in self.motors.items():
                self.control.refresh_motor_status(motor)
                if key == "gripper":
                    obs_dict[f"{key}.pos"] = map_range(
                        motor.getPosition(), self.gripper_open_pos, self.gripper_closed_pos, 0.0, 1.0)
                else:
                    obs_dict[f"{key}.pos"] = motor.getPosition()

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras — use read_latest() for non-blocking reads
        # when the recording loop runs faster than camera FPS
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(
            ".pos"): val for key, val in action.items() if key.endswith(".pos")}

        if self._rt_robot is not None:
            # RT mode: write via seqlock (~1 microsecond, thread-safe)
            motor_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
            q_des = np.array([goal_pos.get(name, 0.0) for name in motor_names], dtype=np.float64)
            self._rt_robot.command_joint_pos(q_des)
            if "gripper" in goal_pos:
                self._rt_robot.command_gripper(float(goal_pos["gripper"]))
        else:
            # Python serial mode
            for key, motor in self.motors.items():
                if key == "gripper":
                    self.control.refresh_motor_status(motor)
                    gripper_goal_pos_mapped = map_range(goal_pos[key], 0.0, 1.0, self.gripper_open_pos, self.gripper_closed_pos)
                    self.control.control_pos_force(motor, gripper_goal_pos_mapped, self.DM4310_SPEED*self.EMIT_VELOCITY_SCALE,
                                                   i_des=self.config.max_gripper_torque/self.DM4310_TORQUE_CONSTANT*self.EMIT_CURRENT_SCALE)
                else:
                    if key in self.JOINT_LIMITS:
                        goal_pos[key] = np.clip(goal_pos[key], self.JOINT_LIMITS[key][0], self.JOINT_LIMITS[key][1])

                    self.control.control_Pos_Vel(
                        motor, goal_pos[key], self.config.joint_velocity_scaling*self.DM4340_SPEED)

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._rt_robot is not None:
            self._rt_robot.disconnect()
            self._rt_robot = None
        elif self.control is not None:
            if self.config.disable_torque_on_disconnect:
                for motor in self.motors.values():
                    self.control.disable(motor)
            else:
                self.control.serial_.close()
        self.bus_connected = False

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
