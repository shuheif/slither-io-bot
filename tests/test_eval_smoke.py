"""End-to-end smoke of the eval harness on short sim episodes."""

import json

import pytest

from slither_bot.eval.harness import evaluate, split_params
from slither_bot.sim.backend import SimBackend
from slither_bot.sim.world import SimConfig


@pytest.mark.smoke
def test_evaluate_three_planners(tmp_path):
    backend = SimBackend(SimConfig(), seed=0)
    payload = evaluate(
        ["random", "apf", "cbf"],
        episodes=2,
        base_seed=1,
        max_time=20.0,
        backend=backend,
        out_dir=tmp_path,
        render=False,
    )

    summary = payload["summary"]
    assert set(summary) == {"random", "apf", "cbf"}
    for agg in summary.values():
        assert agg["episodes"] == 2
        assert 0.0 <= agg["timeout_rate"] <= 1.0
        assert agg["survival_mean"] <= 20.5

    # Paired seeds, safety filter on: cbf must not lose to the random baseline.
    assert summary["cbf"]["survival_mean"] >= summary["random"]["survival_mean"]

    json_files = list(tmp_path.glob("eval_*.json"))
    assert len(json_files) == 1
    stored = json.loads(json_files[0].read_text())
    assert stored["config"]["episodes"] == 2
    assert set(stored["episodes_detail"]) == {"random", "apf", "cbf"}
    assert (tmp_path / "eval_table.md").read_text().count("|") > 10


def test_max_time_zero_disables_the_cap():
    from slither_bot.eval.runner import run_episode
    from slither_bot.planners import make_planner

    # Uncapped episode ends by death, never by timeout (random dies fast).
    backend = SimBackend(SimConfig(), seed=1)
    result = run_episode(backend, make_planner("random", seed=1), "random", 1, max_time=0.0)
    assert result.cause != "timeout"
    assert result.survival_time > 0.0


def test_split_params_prefixes():
    per = split_params({"cbf.gamma": "2.0"}, ["cbf", "apf"])
    assert per == {"cbf": {"gamma": "2.0"}, "apf": {}}


def test_split_params_unprefixed_requires_single_planner():
    assert split_params({"gamma": "2.0"}, ["cbf"]) == {"cbf": {"gamma": "2.0"}}
    with pytest.raises(SystemExit):
        split_params({"gamma": "2.0"}, ["cbf", "apf"])
