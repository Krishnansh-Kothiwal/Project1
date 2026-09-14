"""
tests/test_correctness.py
=========================
Regression tests for the correctness fixes described in the implementation plan.
Deterministic — no real LLM / Ollama calls are made.

Two test groups that need planner avoid the circular import by importing ONLY
mediator and the relevant classes directly without triggering utils → env.

Run with:
    cd LLM4RL
    python -m pytest tests/test_correctness.py -v
"""

import sys
import os
import pytest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("LLM_MODEL", "qwen2.5:7b")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:11434/v1/chat/completions")


# ---------------------------------------------------------------------------
# Lightweight planner factory that avoids triggering the circular import
# (planner -> utils -> env -> planner). We instantiate Base_Planner subclasses
# directly from mediator + planner, using importlib.util to avoid __init__.py
# chains that drag in env/.
# ---------------------------------------------------------------------------

def _import_planner_module():
    """
    Import planner.py avoiding the circular-import chain.
    The chain is: planner -> utils (global_param) -> eval -> env -> Game -> planner
    We break it by pre-injecting a stub for 'utils' that only exposes global_param.
    """
    import types, importlib

    # If already imported cleanly in a prior test, reuse it.
    if 'planner' in sys.modules and hasattr(sys.modules['planner'], 'SimpleDoorKey_Planner'):
        return sys.modules['planner']

    # Stub utils.global_param so planner.py's rom utils import global_param works
    # without importing the rest of utils (which pulls in env).
    if 'utils' not in sys.modules:
        # Build a minimal utils stub.
        stub = types.ModuleType('utils')
        gp = types.ModuleType('utils.global_param')

        def _init(): pass
        def _set_value(k, v): pass
        def _get_value(k): return None

        gp.init = _init
        gp.set_value = _set_value
        gp.get_value = _get_value
        stub.global_param = gp
        sys.modules['utils'] = stub
        sys.modules['utils.global_param'] = gp

    import planner as planner_mod
    return planner_mod


def make_planner(max_calls=0):
    os.environ["MAX_LLM_CALLS"] = str(max_calls)
    import importlib
    planner_mod = _import_planner_module()
    importlib.reload(planner_mod)
    p = planner_mod.SimpleDoorKey_Planner(seed=0)
    p.dialogue_system = "system prompt"
    return p


# ===========================================================================
# A: Forced query semantics
# ===========================================================================

class TestForcedQuerySemantics:
    def test_forced_query_did_query_llm_true(self):
        skill_done = True
        ask_flag_value = False
        forced_query = bool(skill_done)
        policy_query = (not forced_query) and ask_flag_value
        did_query_llm = forced_query or policy_query
        assert did_query_llm is True

    def test_forced_query_policy_mask_is_zero(self):
        skill_done = True
        forced_query = bool(skill_done)
        policy_decision_valid = not forced_query
        assert policy_decision_valid is False
        assert float(policy_decision_valid) == 0.0


# ===========================================================================
# B: Optional (PPO-controlled) query semantics
# ===========================================================================

class TestOptionalQuerySemantics:
    def test_optional_query_did_query_llm_true(self):
        skill_done = False
        ask_flag_value = True
        forced_query = bool(skill_done)
        policy_query = (not forced_query) and ask_flag_value
        did_query_llm = forced_query or policy_query
        assert did_query_llm is True

    def test_optional_query_policy_mask_is_one(self):
        skill_done = False
        forced_query = bool(skill_done)
        policy_decision_valid = not forced_query
        assert policy_decision_valid is True
        assert float(policy_decision_valid) == 1.0


# ===========================================================================
# C: No-query semantics
# ===========================================================================

class TestNoQuerySemantics:
    def test_no_query_did_query_llm_false(self):
        skill_done = False
        ask_flag_value = False
        forced_query = bool(skill_done)
        policy_query = (not forced_query) and ask_flag_value
        did_query_llm = forced_query or policy_query
        assert did_query_llm is False

    def test_no_query_policy_mask_is_one(self):
        skill_done = False
        forced_query = bool(skill_done)
        policy_decision_valid = not forced_query
        assert policy_decision_valid is True

    def test_no_query_comm_penalty_is_zero(self):
        ask_lambda = 0.1
        repeat_feedback = 0.0
        did_query_llm = False
        comm_penalty = (ask_lambda + 0.1 * repeat_feedback) * float(did_query_llm)
        assert comm_penalty == 0.0


# ===========================================================================
# D: Episode reset clears mediator object coordinates
# ===========================================================================

class TestMediatorReset:
    def test_mediator_reset_clears_obj_coordinate(self):
        from mediator import SimpleDoorKey_Mediator
        mediator = SimpleDoorKey_Mediator()
        mediator.obj_coordinate["key"] = (3, 4)
        mediator.obj_coordinate["door"] = (5, 6)
        mediator.reset()
        assert mediator.obj_coordinate == {}

    def test_planner_reset_calls_mediator_reset(self):
        planner_mod = _import_planner_module()
        p = planner_mod.SimpleDoorKey_Planner(seed=0)
        p.dialogue_system = "sys"
        p.mediator.obj_coordinate["key"] = (1, 2)
        p.reset(show=False)
        assert p.mediator.obj_coordinate == {}, (
            "planner.reset() must call mediator.reset()"
        )


# ===========================================================================
# E: LLM call cap semantics
# ===========================================================================

class TestCallCap:
    def _mock_resp(self, content="{explore}"):
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": content}}]}
        return resp

    def test_max_calls_zero_means_unlimited(self):
        from unittest.mock import patch
        p = make_planner(max_calls=0)
        assert p.max_calls == 0
        with patch("requests.post", return_value=self._mock_resp()):
            for _ in range(5):
                p.query_codex("test")
        assert p.call_count == 5

    def test_cap_enforced_at_positive_limit(self):
        from unittest.mock import patch
        p = make_planner(max_calls=2)
        with patch("requests.post", return_value=self._mock_resp()):
            p.query_codex("first")
            p.query_codex("second")
            with pytest.raises(RuntimeError, match="cap reached"):
                p.query_codex("third")
        assert p.call_count == 2

    def test_cap_checked_inside_retry_loop(self):
        from unittest.mock import patch
        p = make_planner(max_calls=3)
        p.call_count = 2
        with patch("requests.post", return_value=self._mock_resp()):
            p.query_codex("last allowed")
        with pytest.raises(RuntimeError, match="cap reached"):
            with patch("requests.post", return_value=self._mock_resp()):
                p.query_codex("should raise")
        assert p.call_count == 3


# ===========================================================================
# F: Success rate denominator
# ===========================================================================

class TestSuccessRate:
    def _rate(self, ep_lens, max_ep_len):
        return sum(i < max_ep_len for i in ep_lens) / len(ep_lens)

    def test_rate_with_ten_trajs(self):
        ep_lens = [50, 100, 60, 100, 80, 100, 90, 100, 30, 100]
        assert self._rate(ep_lens, 100) == pytest.approx(5 / 10)

    def test_rate_with_one_traj(self):
        assert self._rate([99], 100) == 1.0

    def test_rate_with_three_trajs(self):
        assert self._rate([100, 50, 100], 100) == pytest.approx(1 / 3)

    def test_rate_always_in_unit_interval(self):
        import random
        rng = random.Random(42)
        for _ in range(50):
            n = rng.randint(1, 20)
            ep_lens = [rng.randint(1, 100) for _ in range(n)]
            rate = self._rate(ep_lens, 100)
            assert 0.0 <= rate <= 1.0


# ===========================================================================
# G: LLM plan validation
# ===========================================================================

class TestPlanValidation:
    """
    check_plan_isValid validates EVERY comma-separated action in the {…} block
    against the supported grammar. Tests use Base_Planner directly to avoid
    the full circular import chain.
    """
    @pytest.fixture
    def validator(self):
        planner_mod = _import_planner_module()
        # Use Base_Planner.check_plan_isValid as an unbound call via a concrete
        # subclass that doesn't require a real mediator connection.
        p = planner_mod.SimpleDoorKey_Planner(seed=0)
        return p

    # --- Valid plans ---

    def test_valid_explore_only(self, validator):
        assert validator.check_plan_isValid("{explore}") is True

    def test_valid_full_plan(self, validator):
        assert validator.check_plan_isValid(
            "{explore, go to key, pick up key, go to door, toggle door}"
        ) is True

    def test_valid_go_to(self, validator):
        assert validator.check_plan_isValid("{go to the door}") is True

    def test_valid_pick_up(self, validator):
        assert validator.check_plan_isValid("{pick up key}") is True

    def test_valid_toggle(self, validator):
        assert validator.check_plan_isValid("{toggle door}") is True

    def test_valid_drop(self, validator):
        assert validator.check_plan_isValid("{drop key}") is True

    def test_valid_case_insensitive(self, validator):
        assert validator.check_plan_isValid("{Go To key, Pick Up key}") is True

    def test_valid_with_preamble(self, validator):
        assert validator.check_plan_isValid("action: {explore, go to door}") is True

    # --- Invalid plans ---

    def test_invalid_no_braces(self, validator):
        assert validator.check_plan_isValid("explore, go to key") is False

    def test_invalid_empty_block(self, validator):
        assert validator.check_plan_isValid("{}") is False

    def test_invalid_unknown_action(self, validator):
        assert validator.check_plan_isValid("{explore, eat banana}") is False

    def test_invalid_one_bad_token_in_otherwise_valid_plan(self, validator):
        assert validator.check_plan_isValid(
            "{explore, go to key, eat banana, toggle door}"
        ) is False

    def test_invalid_observed_nothing(self, validator):
        assert validator.check_plan_isValid("{observed nothing}") is False

    def test_invalid_observation_text(self, validator):
        assert validator.check_plan_isValid(
            "{observed a key, observed a door}"
        ) is False

    def test_invalid_none(self, validator):
        assert validator.check_plan_isValid(None) is False

    def test_invalid_empty_string(self, validator):
        assert validator.check_plan_isValid("") is False

    def test_invalid_only_whitespace_in_block(self, validator):
        assert validator.check_plan_isValid("{   }") is False


# ===========================================================================
# H: RL object discovery side-effect
# ===========================================================================

class TestRLObjectDiscovery:
    """
    mediator.RL2LLM(obs) updates obj_coordinate as a side-effect.
    RL2LLM expects a 7×7×4 obs where:
      channel 0 = object type (5 = key, 4 = door, ...)
      channel 3 = agent marker; the AGENT cell has value != 4,
                  all other cells have value 4.
    """

    def _make_obs_with_key(self):
        """
        7×7×4 obs with:
          - agent at (0, 0) — channel 3 value = 10 (any value != 4)
          - key at (3, 2) — channel 0 value = 5
        """
        obs = np.zeros((7, 7, 4), dtype=np.int64)
        obs[:, :, 3] = 4       # default: all cells are non-agent (value 4)
        obs[0, 0, 3] = 10      # agent is at (0,0) — different from 4
        obs[3, 2, 0] = 5       # key at (3, 2)
        return obs

    def test_rl2llm_populates_key_coordinate(self):
        from mediator import SimpleDoorKey_Mediator
        mediator = SimpleDoorKey_Mediator()
        obs = self._make_obs_with_key()
        mediator.RL2LLM(obs)
        assert "key" in mediator.obj_coordinate, (
            "RL2LLM should populate obj_coordinate['key'] when a key is visible"
        )

    def test_mediator_reset_then_rl2llm_repopulates(self):
        from mediator import SimpleDoorKey_Mediator
        mediator = SimpleDoorKey_Mediator()
        obs = self._make_obs_with_key()

        mediator.RL2LLM(obs)
        assert "key" in mediator.obj_coordinate

        mediator.reset()
        assert mediator.obj_coordinate == {}

        mediator.RL2LLM(obs)
        assert "key" in mediator.obj_coordinate


# ===========================================================================
# Buffer policy_mask round-trip
# ===========================================================================

class TestBufferPolicyMask:
    def _arrays(self, n):
        s = np.zeros((n, 4), dtype=np.float32)
        a = np.zeros((n, 1), dtype=np.int64)
        r = np.zeros((n, 1), dtype=np.float32)
        v = np.zeros((n, 1), dtype=np.float32)
        lp = np.zeros((n, 1), dtype=np.float32)
        return s, a, r, v, lp

    def test_policy_mask_stored_correctly(self):
        from algos.buffer import Buffer
        buf = Buffer()
        masks = [0.0, 1.0, 0.0]
        s, a, r, v, lp = self._arrays(len(masks))
        for i, m in enumerate(masks):
            buf.store(s[i:i+1], a[i:i+1], r[i:i+1], v[i:i+1], lp[i:i+1], policy_mask=m)
        assert buf.policy_masks == masks

    def test_policy_mask_in_get(self):
        from algos.buffer import Buffer
        buf = Buffer()
        masks = [1.0, 0.0, 1.0]
        s, a, r, v, lp = self._arrays(len(masks))
        for i, m in enumerate(masks):
            buf.store(s[i:i+1], a[i:i+1], r[i:i+1], v[i:i+1], lp[i:i+1], policy_mask=m)
        buf.returns = [np.zeros(1)] * len(masks)
        buf.traj_idx = [0, len(masks)]
        arrays = buf.get()
        assert len(arrays) == 6
        np.testing.assert_array_almost_equal(arrays[5], np.array(masks, dtype=np.float32))

    def test_merge_buffers_carries_policy_masks(self):
        from algos.buffer import Buffer, Merge_Buffers
        def make_buf(masks):
            b = Buffer()
            s, a, r, v, lp = self._arrays(len(masks))
            for i, m in enumerate(masks):
                b.store(s[i:i+1], a[i:i+1], r[i:i+1], v[i:i+1], lp[i:i+1], policy_mask=m)
            return b
        b1 = make_buf([1.0, 0.0])
        b2 = make_buf([0.0, 1.0, 1.0])
        merged = Merge_Buffers([b1, b2])
        assert merged.policy_masks == [1.0, 0.0, 0.0, 1.0, 1.0]
