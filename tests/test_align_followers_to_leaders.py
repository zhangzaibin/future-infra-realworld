"""Hardware-free checks for the standalone alignment script."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location(
    "align_followers", Path(__file__).resolve().parents[1] / "scripts/align_followers_to_leaders.py"
)
aligner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aligner)


class Bus:
    def __init__(self, value):
        names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
        self.motors = {name: SimpleNamespace(id=i) for i, name in enumerate(names, 1)}
        self.positions = dict.fromkeys(names, value)
        self.is_connected = False
        self.writes = []
        self.events = []
        self.stalled = False

    def connect(self):
        self.is_connected = True

    def disconnect(self, disable_torque=True):
        self.events.append(("disconnect", disable_torque))
        self.is_connected = False

    def disable_torque(self):
        self.events.append("disable")

    def sync_read(self, register, **kwargs):
        assert register == "Present_Position"
        return dict(self.positions)

    def sync_write(self, register, positions, **kwargs):
        assert register == "Goal_Position"
        self.events.append("write")
        self.writes.append(dict(positions))
        if not self.stalled:
            self.positions.update(positions)


def arm(port, value):
    bus = Bus(value)
    return SimpleNamespace(
        bus=bus, id=port, config=SimpleNamespace(port=port), is_calibrated=True,
        calibration_fpath=Path("unused.json"),
        calibration={
            name: SimpleNamespace(id=motor.id, range_min=0, range_max=4095)
            for name, motor in bus.motors.items()
        },
        configure=Mock(side_effect=lambda: bus.events.append("configure")),
    )


@pytest.fixture
def setup(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(aligner.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(aligner.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    pairs = [(arm("0", 0), arm("1", 60)), (arm("2", 10), arm("3", 30))]
    return aligner.parse_args([]), pairs


def test_both_pairs_reach_their_own_target_with_bounded_steps(setup):
    args, pairs = setup
    aligner.align(args, pairs)
    for follower, leader in pairs:
        error = max(abs(follower.bus.positions[k] - v) for k, v in leader.bus.positions.items())
        assert error <= args.tolerance
        assert leader.bus.writes == []
        assert leader.bus.events == [("disconnect", False)]
        assert follower.bus.events[:3] == ["disable", "write", "configure"]
        assert follower.bus.events[-1] == ("disconnect", False)
        for previous, current in zip(follower.bus.writes, follower.bus.writes[1:]):
            assert max(abs(current[k] - previous[k]) for k in current) <= args.max_step + 1e-9


def test_dry_run_does_not_write_or_change_torque(setup):
    args, pairs = setup
    args.dry_run = True
    aligner.align(args, pairs)
    for pair in pairs:
        for device in pair:
            assert device.bus.writes == []
            assert device.bus.events == [("disconnect", False)]
            device.configure.assert_not_called()


def test_stalled_arm_times_out_and_releases_both_followers(setup):
    args, pairs = setup
    pairs[0][0].bus.stalled = True
    with pytest.raises(TimeoutError):
        aligner.align(args, pairs)
    assert max(pairs[0][0].bus.writes[-1].values()) <= args.max_step
    for follower, leader in pairs:
        assert follower.bus.events[-1] == ("disconnect", True)
        assert leader.bus.events[-1] == ("disconnect", False)


def test_release_on_success(setup):
    args, pairs = setup
    args.release_on_exit = True
    aligner.align(args, pairs)
    assert all(follower.bus.events[-1] == ("disconnect", True) for follower, _ in pairs)


def test_calibration_mismatch_never_activates_followers(setup):
    args, pairs = setup
    pairs[1][1].is_calibrated = False
    with pytest.raises(ValueError, match="电机校准"):
        aligner.align(args, pairs)
    for pair in pairs:
        for device in pair:
            assert not device.bus.is_connected
            assert device.bus.writes == []
            device.configure.assert_not_called()


def test_partial_connection_failure_closes_open_ports(setup):
    args, pairs = setup

    def fail_connect():
        pairs[1][0].bus.is_connected = True
        raise ConnectionError("handshake failed")

    pairs[1][0].bus.connect = fail_connect
    with pytest.raises(ConnectionError):
        aligner.align(args, pairs)
    assert all(not device.bus.is_connected for pair in pairs for device in pair)


def test_interrupt_releases_activated_followers(setup):
    args, pairs = setup
    pairs[1][0].configure.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        aligner.align(args, pairs)
    assert all(follower.bus.events[-1] == ("disconnect", True) for follower, _ in pairs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 101, -101])
def test_invalid_positions_abort_before_motor_writes(setup, value):
    args, pairs = setup
    pairs[1][1].bus.positions["elbow_flex"] = value
    with pytest.raises(ValueError, match="位置无效"):
        aligner.align(args, pairs)
    assert all(not device.bus.writes for pair in pairs for device in pair)


@pytest.mark.parametrize("argv", [
    ["--duration", "nan"], ["--fps", "0"], ["--max-step", "-1"],
    ["--tolerance", "inf"], ["--timeout", "1"], ["--left-leader-port", "/dev/ttyACM0"],
])
def test_invalid_arguments(argv):
    with pytest.raises(SystemExit):
        aligner.parse_args(argv)


@pytest.mark.parametrize("dry_run", [False, True])
def test_disconnect_failure_is_reported_and_other_ports_are_closed(setup, dry_run):
    args, pairs = setup
    args.dry_run = dry_run
    failing_bus = pairs[1][0].bus
    original = failing_bus.disconnect
    attempts = []

    def fail_once(disable_torque=True):
        attempts.append(disable_torque)
        if len(attempts) == 1:
            raise OSError("disconnect failed")
        original(disable_torque)

    failing_bus.disconnect = fail_once
    with pytest.raises(RuntimeError, match="断开机械臂失败"):
        aligner.align(args, pairs)
    assert len(attempts) == 2
    assert all(not device.bus.is_connected for pair in pairs for device in pair)


def test_missing_calibration_does_not_open_any_ports(setup):
    args, pairs = setup
    pairs[1][1].calibration = {}
    with pytest.raises(ValueError, match="缺少完整校准"):
        aligner.align(args, pairs)
    assert all(not device.bus.events for pair in pairs for device in pair)
