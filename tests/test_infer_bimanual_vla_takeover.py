from threading import Event
from unittest.mock import MagicMock, patch

import pytest

from lerobot.processor import make_default_processors
from lerobot.scripts.infer_bimanual_vla import (
    STOP_DURATION,
    STOP_SHUTDOWN,
    STOP_TAKEOVER,
    STOP_USER,
    BimanualVLAInferConfig,
    TakeoverDatasetConfig,
    dataset_meta_exists,
    inference_stop_reason,
    record_takeover_episode,
    run_inference_loop,
    setup_takeover_dataset,
    wait_for_vla_resume,
)
from lerobot.scripts.lerobot_record import record_loop as real_record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobot, MockRobotConfig
from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig


def test_takeover_requires_both_teleop_and_dataset():
    with pytest.raises(ValueError, match="人工接管"):
        BimanualVLAInferConfig(robot=MockRobotConfig(), teleop=MockTeleopConfig())

    with pytest.raises(ValueError, match="人工接管"):
        BimanualVLAInferConfig(
            robot=MockRobotConfig(),
            dataset=TakeoverDatasetConfig(repo_id=DUMMY_REPO_ID),
        )


def test_takeover_dataset_defaults_task_and_fps():
    cfg = BimanualVLAInferConfig(
        robot=MockRobotConfig(),
        task="fold the towels",
        fps=30,
        teleop=MockTeleopConfig(),
        dataset=TakeoverDatasetConfig(repo_id=DUMMY_REPO_ID),
    )
    assert cfg.enable_takeover
    assert cfg.dataset.single_task == "fold the towels"
    assert cfg.dataset.fps == 30
    assert cfg.dataset.push_to_hub is False


def test_inference_stop_reason():
    import time

    shutdown = Event()
    start = time.perf_counter()
    assert inference_stop_reason(shutdown, None, start, duration=10) is None

    events = {"takeover": False, "stop_recording": False}
    assert inference_stop_reason(shutdown, events, start, duration=10) is None

    events["takeover"] = True
    assert inference_stop_reason(shutdown, events, start, duration=10) == STOP_TAKEOVER

    events["takeover"] = False
    events["stop_recording"] = True
    assert inference_stop_reason(shutdown, events, start, duration=10) == STOP_USER

    shutdown.set()
    assert (
        inference_stop_reason(shutdown, {"takeover": True, "stop_recording": False}, start, 10)
        == STOP_SHUTDOWN
    )

    shutdown = Event()
    assert (
        inference_stop_reason(shutdown, None, start_time=time.perf_counter() - 20, duration=10)
        == STOP_DURATION
    )


def test_run_inference_loop_returns_takeover_without_stepping():
    reason = run_inference_loop(
        robot=MagicMock(),
        vla_client=MagicMock(),
        obs_adapter=MagicMock(),
        action_adapter=MagicMock(),
        action_limiter=None,
        shutdown_event=Event(),
        task="fold the towels",
        fps=30,
        duration=10,
        events={"takeover": True, "stop_recording": False},
    )
    assert reason == STOP_TAKEOVER


def test_keyboard_listener_includes_takeover_event():
    with patch("lerobot.utils.control_utils.is_headless", return_value=True):
        listener, events = init_keyboard_listener(manual_episode_control=True, enable_takeover=True)
    assert listener is None
    assert events["takeover"] is False
    assert events["recording"] is False
    assert events["confirm_episode"] is False
    assert events["discard_episode"] is False


def test_dataset_meta_exists(tmp_path):
    assert dataset_meta_exists(None) is False
    assert dataset_meta_exists(tmp_path) is False
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("{}")
    assert dataset_meta_exists(tmp_path) is True


def _make_takeover_dataset(tmp_path):
    robot = MockRobot(MockRobotConfig(calibration_dir=tmp_path / "robot_calibration"))
    robot.connect()
    teleop_action_processor, _, robot_observation_processor = make_default_processors()
    dataset = setup_takeover_dataset(
        robot=robot,
        dataset_cfg=TakeoverDatasetConfig(
            repo_id=DUMMY_REPO_ID,
            single_task="Dummy task",
            root=tmp_path / "takeover",
            num_episodes=2,
            push_to_hub=False,
        ),
        teleop_action_processor=teleop_action_processor,
        robot_observation_processor=robot_observation_processor,
        resume=True,
    )
    return robot, dataset, teleop_action_processor, robot_observation_processor


def test_setup_takeover_dataset_create_and_resume(tmp_path):
    robot, dataset, teleop_action_processor, robot_observation_processor = _make_takeover_dataset(tmp_path)
    assert dataset.num_episodes == 0
    assert dataset_meta_exists(tmp_path / "takeover")

    with pytest.raises(ValueError, match="already exists"):
        setup_takeover_dataset(
            robot=robot,
            dataset_cfg=TakeoverDatasetConfig(
                repo_id=DUMMY_REPO_ID,
                single_task="Dummy task",
                root=tmp_path / "takeover",
                push_to_hub=False,
            ),
            teleop_action_processor=teleop_action_processor,
            robot_observation_processor=robot_observation_processor,
            resume=False,
        )


def test_record_takeover_episode_discards_then_saves(tmp_path):
    robot, dataset, teleop_action_processor, robot_observation_processor = _make_takeover_dataset(tmp_path)
    teleop = MockTeleop(MockTeleopConfig(calibration_dir=tmp_path / "teleop_calibration"))
    teleop.connect()
    _, robot_action_processor, _ = make_default_processors()

    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "start_episode": False,
        "confirm_episode": False,
        "discard_episode": False,
        "recording": False,
        "takeover": True,
    }
    decisions = iter(("discard_episode", "confirm_episode"))

    def run_short_episode(**kwargs):
        kwargs["control_time_s"] = 0.1
        real_record_loop(**kwargs)
        events[next(decisions)] = True

    with patch("lerobot.scripts.infer_bimanual_vla.record_loop", side_effect=run_short_episode):
        discarded = record_takeover_episode(
            robot=robot,
            teleop=teleop,
            dataset=dataset,
            events=events,
            fps=dataset.fps,
            single_task="Dummy task",
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )
        saved = record_takeover_episode(
            robot=robot,
            teleop=teleop,
            dataset=dataset,
            events=events,
            fps=dataset.fps,
            single_task="Dummy task",
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

    assert discarded == "discarded"
    assert saved == "saved"
    assert dataset.num_episodes == 1


def test_wait_for_vla_resume_stops_on_escape():
    events = {
        "recording": True,
        "start_episode": False,
        "exit_early": True,
        "takeover": True,
        "stop_recording": True,
    }
    assert wait_for_vla_resume(events, Event()) is False
    assert events["takeover"] is False
    assert events["recording"] is False
