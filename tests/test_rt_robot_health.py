"""DK1RobotRT.health_problems() against a fake native loop (no hardware, no build).

Covers the 2026-09-30 wedge: a loop whose cycle counter stops (the RT thread is
blocked on the bus) and a latched TX stall must both be reported, in the
"<fact> — <fix>" shape the rollout's failure parser splits on, without an arm
prefix (the bimanual follower adds "left arm: " / "right arm: ").
"""

from types import SimpleNamespace

import pytest

from trlc_dk1_control import rt_robot
from trlc_dk1_control.config import DK1RobotConfig
from trlc_dk1_control.rt_robot import DK1RobotRT


class FakeLoop:
    def __init__(self):
        self.loop_count = 1000
        self.running = True
        self.comm_loss = False
        self.tx_stalled = False
        self.stop_result = True

    def get_health(self):
        return SimpleNamespace(
            loop_count=self.loop_count,
            comm_loss=self.comm_loss,
            tx_stalled=self.tx_stalled,
            motor_stale=[False] * 7,
            damping_mode=False,
            overcurrent_count=0,
            overspeed_count=0,
        )

    def is_running(self):
        return self.running

    def stop(self):
        self.running = False
        return self.stop_result


@pytest.fixture
def robot():
    r = object.__new__(DK1RobotRT)
    r._config = DK1RobotConfig(label="left")
    r._rt_cfg = SimpleNamespace(max_consecutive_empty_cycles=50, max_consecutive_tx_fail_cycles=50)
    r._loop = FakeLoop()
    r._last_progress = None
    return r


def _one_separator(msg: str) -> bool:
    return msg.count(" — ") == 1 and not msg.startswith(("left", "right"))


def test_healthy_loop_reports_nothing(robot):
    assert robot.health_problems() == []
    robot._loop.loop_count += 60
    assert robot.health_problems() == []


def test_stalled_cycle_counter_is_reported(robot, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(rt_robot.time, "monotonic", lambda: clock[0])
    assert robot.health_problems() == []  # first read only records progress
    clock[0] += 0.1
    assert robot.health_problems() == []  # not yet past the stall threshold
    clock[0] += 0.4
    (problem,) = robot.health_problems()
    assert problem.startswith("RT control loop stalled (no cycles for 0.5 s)")
    assert _one_separator(problem)
    robot._loop.loop_count += 1  # it moves again -> healthy
    assert robot.health_problems() == []


def test_tx_stall_message(robot):
    robot._loop.comm_loss = True
    robot._loop.tx_stalled = True
    robot._loop.loop_count += 1
    (problem,) = robot.health_problems()
    assert problem.startswith("CAN TX stalled")
    assert "CAN cable/connectors and motor power" in problem
    assert _one_separator(problem)


def test_rx_comm_loss_keeps_its_message(robot):
    robot._loop.comm_loss = True
    (problem,) = robot.health_problems()
    assert problem.startswith("motor-bus comm loss")
    assert _one_separator(problem)


def test_abandoned_loop_is_kept_alive(robot):
    loop = robot._loop
    loop.stop_result = False
    robot.disconnect()
    assert robot._loop is None
    assert loop in rt_robot._ABANDONED_LOOPS
    rt_robot._ABANDONED_LOOPS.remove(loop)
