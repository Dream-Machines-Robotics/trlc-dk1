"""
DK1RobotRT — Drop-in replacement for DK1Robot using the C++ RT control loop.

Uses the _trlc_dk1_rt nanobind extension for a single-threaded 250 Hz
control loop with optional PREEMPT_RT support and lock-free communication.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import numpy as np

from .config import DK1RobotConfig

logger = logging.getLogger(__name__)

# Default motor configuration matching motor_chain.py
_DEFAULT_MOTORS = [
    {"name": "joint_1", "type": "DM4340",  "slave_id": 0x01, "master_id": 0x11},
    {"name": "joint_2", "type": "DM4340",  "slave_id": 0x02, "master_id": 0x12},
    {"name": "joint_3", "type": "DM4340",  "slave_id": 0x03, "master_id": 0x13},
    {"name": "joint_4", "type": "DM4310",  "slave_id": 0x04, "master_id": 0x14},
    {"name": "joint_5", "type": "DM4310",  "slave_id": 0x05, "master_id": 0x15},
    {"name": "joint_6", "type": "DM4310",  "slave_id": 0x06, "master_id": 0x16},
    {"name": "gripper",  "type": "DM4310",  "slave_id": 0x07, "master_id": 0x17},
]


class DK1RobotRT:
    """
    Drop-in replacement for DK1Robot using the C++ RT control loop.

    Provides the same public API as DK1Robot for use with DK1Follower
    and other orchestration code.
    """

    def __init__(self, config: DK1RobotConfig) -> None:
        try:
            from trlc_dk1_control._trlc_dk1_rt import RtControlLoop, RtLoopConfig, MotorDescriptor, MotorType
        except ImportError as e:
            raise ImportError(
                "C++ RT extension not available. Install with: "
                "pip install -e '.[rt]'"
            ) from e

        self._config = config
        self._RtControlLoop = RtControlLoop
        self._loop = None
        self._warned_no_accel_guard = False

        # Build RtLoopConfig from DK1RobotConfig
        rt_cfg = RtLoopConfig()
        rt_cfg.serial_port = config.serial_port
        rt_cfg.loop_hz = config.motor_thread_hz

        # Motor descriptors
        motor_type_map = {
            "DM4310": MotorType.DM4310,
            "DM4310_48V": MotorType.DM4310_48V,
            "DM4340": MotorType.DM4340,
            "DM4340_48V": MotorType.DM4340_48V,
        }

        motors = []
        for m in _DEFAULT_MOTORS:
            desc = MotorDescriptor()
            desc.name = m["name"]
            desc.type = motor_type_map[m["type"]]
            desc.slave_id = m["slave_id"]
            desc.master_id = m["master_id"]
            motors.append(desc)
        rt_cfg.motors = motors

        # Gains
        rt_cfg.default_kp = np.asarray(config.arm_kp, dtype=np.float64)
        rt_cfg.default_kd = np.asarray(config.arm_kd, dtype=np.float64)

        # Joint limits (flatten from (6,2) to (12,) interleaved [lo0,hi0,...])
        limits = np.asarray(config.joint_pos_limits, dtype=np.float64)
        rt_cfg.joint_pos_limits = limits.flatten()

        rt_cfg.joint_torque_limits = np.asarray(config.joint_torque_limits, dtype=np.float64)

        # joint_velocity_scaling caps COMMANDED motion only — the per-cycle slew
        # limit (rad/cycle, the impedance loop's position ramp). It deliberately
        # does NOT scale the over-speed SAFETY limit (joint_velocity_limits,
        # rad/s, used by the damping watchdog).
        #
        # Why decoupled: coupling them meant a cautious low scaling also tightened
        # the over-speed trip — at 0.2 the limit became ~1 rad/s, which normal
        # leader teleop exceeds, so the watchdog slammed into damping mode and the
        # arm sagged (kp→0 + gravity-comp off). The slew cap already bounds how
        # fast the loop can *command* the arm, so the over-speed watchdog stays at
        # the physical limit as a true backstop for instability / external push /
        # motor runaway. See agent_skills/eval-safety.md.
        vel_scale = float(np.clip(config.joint_velocity_scaling, 1e-3, 1.0))
        if vel_scale != 1.0:
            logger.info(
                "DK1RobotRT: joint_velocity_scaling=%.3f → scaling max_pos_delta_per_cycle "
                "(commanded slew) only; over-speed limit kept at physical max.",
                vel_scale,
            )
        rt_cfg.joint_velocity_limits = np.asarray(config.joint_velocity_limits, dtype=np.float64)
        rt_cfg.max_pos_delta_per_cycle = np.asarray(config.max_pos_delta_per_cycle, dtype=np.float64) * vel_scale
        # Acceleration guard: rad/s² → rad/cycle² at the RT loop rate. Deliberately
        # NOT multiplied by vel_scale — it bounds how fast the slew ramp may build
        # up, independent of how fast it is allowed to go.
        accel_limits = np.asarray(config.joint_accel_limits, dtype=np.float64)
        if np.any(accel_limits > 0.0):
            logger.info(
                "DK1RobotRT: joint_accel_limits=%s rad/s² → acceleration guard on the "
                "commanded slew ramp (0 = off for that joint).",
                np.array2string(accel_limits, precision=1),
            )
        rt_cfg.max_accel_per_cycle2 = accel_limits / (config.motor_thread_hz**2)
        rt_cfg.limit_buffer = 0.05

        # Gravity compensation
        if config.mjcf_path:
            rt_cfg.model_path = str(Path(config.mjcf_path).resolve())
        rt_cfg.gravity_comp_scale = config.gravity_comp_scale

        # Safety
        rt_cfg.command_timeout_s = config.command_timeout_s
        rt_cfg.overcurrent_threshold = config.overcurrent_threshold
        rt_cfg.overspeed_threshold = config.overspeed_threshold
        rt_cfg.min_motors_required = config.min_motors_required
        rt_cfg.gripper_cal_timeout_s = config.gripper_cal_timeout_s
        rt_cfg.max_consecutive_empty_cycles = config.max_consecutive_empty_cycles

        # Gripper
        rt_cfg.gripper_open_pos = config.gripper_open_pos
        rt_cfg.gripper_closed_pos = config.gripper_closed_pos
        rt_cfg.max_gripper_torque_nm = config.max_gripper_torque_nm
        rt_cfg.torque_constant = config.DM4310_TORQUE_CONSTANT
        rt_cfg.emit_velocity_scale = config.EMIT_VELOCITY_SCALE
        rt_cfg.emit_current_scale = config.EMIT_CURRENT_SCALE
        rt_cfg.disable_torque_on_disconnect = config.disable_torque_on_disconnect

        # ④ Optional CPU pinning / priority for the SCHED_FIFO control thread,
        # opt-in via env (defaults keep the C++ values: priority 80, affinity
        # -1 = unpinned). Pinning only buys jitter reduction under contention;
        # for *true* isolation the target core must also be kept clear of other
        # work (kernel `isolcpus=` / a cpuset), which is a system-config step.
        aff = os.environ.get("LEROBOT_RT_CPU_AFFINITY", "").strip()
        if aff:
            try:
                rt_cfg.rt_cpu_affinity = int(aff)
                if rt_cfg.rt_cpu_affinity >= 0:
                    logger.info("DK1RobotRT: pinning control loop to CPU %d", rt_cfg.rt_cpu_affinity)
                else:
                    logger.info("DK1RobotRT: CPU pinning disabled (affinity %d)", rt_cfg.rt_cpu_affinity)
            except ValueError:
                logger.warning("DK1RobotRT: ignoring bad LEROBOT_RT_CPU_AFFINITY=%r", aff)
        prio = os.environ.get("LEROBOT_RT_PRIORITY", "").strip()
        if prio:
            try:
                rt_cfg.rt_priority = int(prio)
                logger.info("DK1RobotRT: control-loop SCHED_FIFO priority %d", rt_cfg.rt_priority)
            except ValueError:
                logger.warning("DK1RobotRT: ignoring bad LEROBOT_RT_PRIORITY=%r", prio)

        self._rt_cfg = rt_cfg

    def connect(self) -> None:
        """Start the C++ RT control loop."""
        self._loop = self._RtControlLoop(self._rt_cfg)
        try:
            self._loop.start()
        except RuntimeError as exc:
            # Turn the C++ "only N/M motors responded" error into a clear,
            # copy-pasteable diagnosis (E-STOP/power vs a single dead motor).
            self._explain_motor_init_failure(exc)
            raise
        # RT scheduling is applied inside the RT thread, AFTER mlockall — which on
        # the FIRST loop can take a second or two while it locks the whole
        # process's pages (torch/mujoco/CUDA). is_rt_active() therefore reads
        # False if polled right after start() (and the second arm's mlockall is
        # fast, so the two arms would disagree — exactly the spurious "RT NOT
        # active" warning). The loop already prints the AUTHORITATIVE status to
        # stderr ("RT scheduling active (SCHED_FIFO priority N)" or "RT scheduling
        # not available, using default scheduler"), so we don't re-derive it here.
        logger.info("DK1RobotRT connected (RT scheduling status printed by the loop)")

    def _explain_motor_init_failure(self, exc: RuntimeError) -> None:
        """Log a human/LM-friendly diagnosis for a motor-init failure.

        Distinguishes the two common causes from the count of motors that
        answered: NONE responding ⇒ E-STOP engaged or the arm's motor power is
        off (the USB-CAN adapter still enumerates on USB power, so the port
        opens but nothing replies); SOME responding ⇒ power reaches the bus, so
        it's a per-motor fault (power/CAN connector or a mis-set CAN ID) on the
        silent one(s). No-op for unrelated RuntimeErrors."""
        msg = str(exc)
        if "motors responded" not in msg and "Motor initialization failed" not in msg:
            return
        port = getattr(self._config, "serial_port", "?")
        m = re.search(r"(\d+)\s*/\s*(\d+)\s+arm motors responded", msg)
        responded, total = (int(m.group(1)), int(m.group(2))) if m else (None, None)
        missing = re.findall(r"MISSING:\s+(\S+)", msg)
        L = ["", "=" * 72, f"  ⛔ ARM CONNECT FAILED — port {port}"]
        if responded is not None:
            L.append(f"     {responded}/{total} motors responded"
                     + (f"; silent: {', '.join(missing)}" if missing else ""))
        if responded == 0:
            L += [
                "     → NONE of the motors answered. The USB-CAN adapter is",
                "       USB-powered, so the serial port still opened — but no motor",
                "       replied. This almost always means one of:",
                "         • the E-STOP is engaged, or",
                "         • this arm's motor power supply (24/48 V) is OFF.",
                "       Fix: release the E-STOP / switch motor power on, then retry.",
            ]
        elif responded and missing:
            L += [
                "     → Some motors answered, so power DOES reach the bus — this",
                "       looks like a per-motor fault on the silent one(s):",
                f"         • check the power + CAN connector for: {', '.join(missing)}",
                "         • reseat the CAN daisy-chain link feeding them",
                "         • verify their CAN IDs (a wrong ID looks like a missing motor)",
            ]
        elif responded is None:
            L += [
                "     → Could not parse the motor count. Raw error:",
                f"       {msg.splitlines()[0]}",
                "       Check the E-STOP, the arm's motor power, and the CAN cable.",
            ]
        L += ["=" * 72, ""]
        logger.error("\n".join(L))

    def disconnect(self) -> None:
        """Stop the C++ RT control loop."""
        if self._loop is not None:
            self._loop.stop()
            self._loop = None
        logger.info("DK1RobotRT disconnected")

    def command_joint_pos(self, q_des: np.ndarray) -> None:
        """Set target joint positions for the 6 arm joints (radians)."""
        if q_des.shape != (6,):
            raise ValueError(f"Expected shape (6,), got {q_des.shape}")
        if self._loop is not None:
            self._loop.command_joint_pos(np.asarray(q_des, dtype=np.float64))

    def command_gripper(self, normalized_pos: float) -> None:
        """Set gripper target (0.0=open, 1.0=closed)."""
        if self._loop is not None:
            self._loop.command_gripper(float(normalized_pos))

    def get_joint_state(self) -> dict[str, np.ndarray]:
        """Return arm joint state as dict with 'pos', 'vel', 'torque' arrays."""
        if self._loop is None:
            return {
                "pos": np.zeros(6),
                "vel": np.zeros(6),
                "torque": np.zeros(6),
            }
        state = self._loop.get_joint_state()
        return {
            "pos": np.array(state.pos),
            "vel": np.array(state.vel),
            "torque": np.array(state.torque),
        }

    def get_gripper_state(self) -> dict[str, float]:
        """Return normalized gripper position and torque."""
        if self._loop is None:
            return {"pos": 0.0, "torque": 0.0}
        state = self._loop.get_gripper_state()
        return {"pos": state.pos, "torque": state.torque}

    def get_state_at(self, ts_ns: int) -> dict | None:
        """Return joint+gripper state at-or-before ``ts_ns``, or ``None`` if unavailable.

        Wraps the C++ ``get_state_at`` ring lookup, flattening the result into a
        dict shaped like ``get_joint_state()`` + ``get_gripper_state()`` combined,
        plus the actual capture timestamp the ring used (so the caller can audit
        how stale the alignment is).

        ``None`` is returned when the requested time predates the oldest sample
        in the ring (~1 s at 250 Hz), or when no samples have been captured yet,
        or when the RT loop isn't running. The caller should then fall back to
        the "latest" API.
        """
        if self._loop is None:
            return None
        # Older native (_trlc_dk1_rt) builds predate this ring-lookup API. Degrade
        # to the "latest" path — follower.get_observation() already falls back to
        # get_joint_state/get_gripper_state when this returns None — instead of
        # raising AttributeError, which killed the 250 Hz state publisher on its
        # first tick and left every episode .rrd empty. (Rebuild the extension to
        # restore at-or-before-ts alignment for OBS_ALIGN=1.)
        get_at = getattr(self._loop, "get_state_at", None)
        if get_at is None:
            return None
        snap = get_at(ts_ns)
        if snap is None:
            return None
        return {
            "ts_mono_ns": snap.ts_mono_ns,
            "joint_pos": np.array(snap.joints.pos),
            "joint_vel": np.array(snap.joints.vel),
            "joint_torque": np.array(snap.joints.torque),
            "gripper_pos": snap.gripper.pos,
            "gripper_torque": snap.gripper.torque,
        }

    def set_accel_guard(self, enabled: bool) -> None:
        """Enable/disable the slew ramp's acceleration guard at runtime.

        Used by the DAgger strategy to bypass the guard while a human
        intervenes (leader teleop is continuous, so the guard only adds lag)
        and restore it before handing control back to the policy. The
        slew-rate cap itself is unaffected.
        """
        if self._loop is None:
            return
        # Older native (_trlc_dk1_rt) builds predate this toggle. Warn once and
        # keep running with the guard permanently on (its compiled-in behavior)
        # rather than raising AttributeError mid-intervention.
        setter = getattr(self._loop, "set_accel_guard", None)
        if setter is None:
            if not self._warned_no_accel_guard:
                logger.warning(
                    "DK1RobotRT: compiled RT extension predates set_accel_guard — "
                    "acceleration guard stays on. Rebuild with `make rt-ext-ensure`."
                )
                self._warned_no_accel_guard = True
            return
        setter(enabled)

    def get_perf(self):
        """Return performance snapshot from the RT loop."""
        if self._loop is None:
            return None
        return self._loop.get_perf()

    def read_cycle_times(self, n: int = 10000) -> np.ndarray:
        """Return array of recent cycle times in microseconds."""
        if self._loop is None:
            return np.array([], dtype=np.float32)
        return self._loop.read_cycle_times(n)
