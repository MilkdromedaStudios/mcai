"""Offline trainer tests: milestones first, no idle wandering or false success."""
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from pyflayer.agent.trainer import Milestone, Trainer
from pyflayer.runtime import Runtime


class TestOvernightTraining(unittest.TestCase):
    @staticmethod
    def trainer():
        t = Trainer.__new__(Trainer)
        t.agent = SimpleNamespace(
            _actuation_lock=threading.RLock(),
            llm=SimpleNamespace(configured=False),
            interrupted_at=0,
            wait_reflexes=Mock(),
        )
        t.brain = SimpleNamespace(milestone=Mock(), milestones=Mock(return_value={}),
                                  remember=Mock())
        t.bot = SimpleNamespace(find_blocks=Mock(return_value=[]))
        t.curriculum = [Milestone("wood", "Find wood", lambda _: False,
                                  lambda _: "said it worked")]
        t.done_this_run = set()
        t.recovery_attempts = {}
        t.running = True
        t.status = "starting"
        t.current = ""
        t.last_error = ""
        t.completed_attempts = 0
        t.successful_attempts = 0
        t.start_time = time.time()
        t.end_time = time.time() + 120
        t.note = Mock()
        t.write_report = Mock()
        t.finish = Mock()
        t._ensure_connected = Mock(return_value=True)
        t._survival_upkeep = Mock()
        return t

    def test_failed_goal_not_skipped_for_exploration(self):
        t = self.trainer()
        wood = t.curriculum[0]
        self.assertIs(t._next_milestone({"wood": 3}), wood)

    def test_failed_goals_retry_with_limited_surveys_then_block(self):
        t = self.trainer()
        t.survey = Mock()
        def fail(goal, fails):
            t.completed_attempts += 1
            fails[goal.name] = fails.get(goal.name, 0) + 1
        t._attempt = fail
        t._practice_loop()
        self.assertEqual(t.status, "blocked")
        self.assertEqual(t.completed_attempts, 12)
        self.assertEqual(t.survey.call_count, 3)
        self.assertTrue(t.last_error)
        t.finish.assert_called_once()

    def test_first_action_is_a_milestone_not_a_random_survey(self):
        t = self.trainer()
        seen = []
        def succeed(goal, fails):
            seen.append(goal.name)
            t.done_this_run.add(goal.name)
            t.successful_attempts += 1
        t._attempt = succeed
        t.survey = Mock()
        t._practice_loop()
        self.assertEqual(seen, ["wood"])
        t.survey.assert_not_called()
        self.assertEqual(t.status, "completed")
        t.finish.assert_called_once()

    def test_success_is_verified_not_assumed_from_planner_string(self):
        t = self.trainer()
        goal = Milestone("stronghold", "Find stronghold", lambda _: False,
                         lambda _: "found it, trust me")
        fails = {}
        t._attempt(goal, fails)
        self.assertEqual(fails["stronghold"], 1)
        self.assertEqual(t.successful_attempts, 0)
        self.assertNotIn("stronghold", t.done_this_run)
        self.assertEqual(t.brain.milestone.call_args.args[1], False)

    def test_unexpected_exception_is_reported_as_error_not_idle(self):
        t = self.trainer()
        t._ensure_connected = Mock(side_effect=RuntimeError("network failed"))
        t._practice_loop()
        self.assertEqual(t.status, "error")
        self.assertIn("network failed", t.last_error)
        t.finish.assert_called_once()

    def test_duplicate_start_training_does_not_spawn_another_trainer(self):
        runtime = Runtime.__new__(Runtime)
        runtime._lock = threading.RLock()
        previous = SimpleNamespace(running=True)
        session = SimpleNamespace(name="PyBot", trainer=previous)
        runtime.session = lambda _: session
        runtime.start_agent = Mock()
        result = runtime.start_training(name="PyBot")
        self.assertIs(result, previous)
        runtime.start_agent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
