"""Tests that need no model. They do need `nu` on the PATH (run them with `mise run test`)."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nushell_lfm.calls import parse_tool_call, wrap_tool_call
from nushell_lfm.core import RUNS_DIR
from nushell_lfm.environments import build_environments, run_nu, same_output
from nushell_lfm.pipeline import build


class Calls(unittest.TestCase):
    def test_round_trip(self):
        code = "ls | where size > 1kb\n| get name"
        self.assertEqual(parse_tool_call(wrap_tool_call(code)), code)

    def test_malformed_call_is_rejected(self):
        self.assertIsNone(parse_tool_call("<function=run_nushell>ls</function>"))
        self.assertIsNone(parse_tool_call(wrap_tool_call("ls") + "\ntrailing text"))

    def test_same_output_ignores_key_order_only(self):
        self.assertTrue(same_output('{"a": 1, "b": 2}', '{"b": 2, "a": 1}'))
        self.assertFalse(same_output("[1, 2]", "[2, 1]"))


class Environments(unittest.TestCase):
    def test_same_seed_same_project(self):
        first, second = build_environments(3), build_environments(3)
        self.assertEqual([e["layout"] for e in first], [e["layout"] for e in second])
        self.assertEqual([e["files"] for e in first], [e["files"] for e in second])

    def test_nu_runs_in_an_environment(self):
        env = build_environments(2)[0]
        result = run_nu("ls | length", env["id"], cache=False)
        self.assertTrue(result["ok"], result["stderr"])
        self.assertGreater(int(result["stdout"]), 0)

    def test_parse_error_is_reported(self):
        env = build_environments(2)[0]
        result = run_nu("ls | where (", env["id"], cache=False)
        self.assertFalse(result["ok"])
        self.assertTrue(result["parse_error"])


class Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.built = build(environments=4, chains=40)
        cls.dataset = cls.built["dataset"]

    def test_every_reference_call_returns_its_value(self):
        for chain in self.built["chains"]:
            result = run_nu(chain["code"], chain["env"])
            self.assertTrue(result["ok"] and same_output(result["stdout"], chain["expected"]), chain["code"])

    def test_held_out_environments_are_whole(self):
        train = {r["env"] for r in self.dataset["target"]["train"]}
        held_out = {r["env"] for r in self.dataset["target"]["dev"]}
        self.assertTrue(held_out)
        self.assertFalse(train & held_out)
        self.assertEqual(sorted(held_out), self.dataset["dev_envs"])

    def test_rows_are_well_formed_calls(self):
        for row in self.dataset["target"]["train"] + self.dataset["target"]["dev"]:
            self.assertIsNotNone(parse_tool_call(row["completion"]))
            self.assertTrue(row["speech"].strip())

    def test_build_is_deterministic(self):
        self.assertEqual(build(environments=4, chains=40)["dataset"]["fingerprint"], self.dataset["fingerprint"])


class SavedResults(unittest.TestCase):
    """Only when a trained adapter is present: its report is for the dataset the defaults build."""

    def test_report_matches_the_default_dataset(self):
        report = RUNS_DIR / "target" / "report.json"
        if not report.exists():
            self.skipTest("no trained adapter in runs/target")
        self.assertEqual(json.loads(report.read_text())["dataset"], build()["dataset"]["fingerprint"])


if __name__ == "__main__":
    unittest.main()
