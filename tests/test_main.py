import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout
from functools import partial
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("week1_main", ROOT / "Week 1" / "main.py")
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)


class FakeChatModel:
    def __init__(self, content="<final_answer>done</final_answer>"):
        self.content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.content)


class ScriptedEngine:
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, messages):
        return next(self.responses)


def tool_request(name, **parameters):
    return '<tool_call>' + json.dumps({"name": name, "parameters": parameters}) + '</tool_call>'


class ModelAdapterTests(unittest.TestCase):
    def test_generate_maps_conversation_roles_and_returns_text(self):
        model = FakeChatModel()
        engine = main.LangChainLLMEngine(model)
        result = engine.generate([
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "previous"},
            {"role": "user", "content": "next"},
        ])
        self.assertEqual(result, "<final_answer>done</final_answer>")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual([type(m) for m in model.calls[0]],
                         [SystemMessage, HumanMessage, AIMessage, HumanMessage])
        self.assertEqual([m.content for m in model.calls[0]],
                         ["rules", "question", "previous", "next"])

    def test_generate_rejects_non_text_response(self):
        engine = main.LangChainLLMEngine(FakeChatModel([{"type": "image"}]))
        with self.assertRaisesRegex(ValueError, "text"):
            engine.generate([{"role": "user", "content": "hello"}])

    def test_configuration_requires_real_api_key(self):
        with patch.dict(os.environ, {"AGENT_API_KEY": "replace-with-your-provider-api-key"}):
            with self.assertRaisesRegex(ValueError, "AGENT_API_KEY"):
                main.LangChainLLMEngine.from_environment(model_factory=lambda **kw: self.fail("constructed"))

    def test_configuration_passes_model_endpoint_and_key(self):
        created = []
        def factory(**kwargs):
            created.append(kwargs)
            return FakeChatModel()
        with patch.dict(os.environ, {"AGENT_API_KEY": "test-key", "AGENT_MODEL": "custom-model",
                                     "AGENT_BASE_URL": "https://example.invalid/v1"}):
            engine = main.LangChainLLMEngine.from_environment(model_factory=factory)
        self.assertIsInstance(engine, main.LangChainLLMEngine)
        self.assertEqual(created, [{"api_key": "test-key", "model": "custom-model",
                                    "base_url": "https://example.invalid/v1"}])

    def test_configuration_uses_current_deepseek_defaults(self):
        created = []
        with patch.dict(os.environ, {"AGENT_API_KEY": "test-key"}, clear=True), \
                patch.object(main, "load_dotenv"):
            main.LangChainLLMEngine.from_environment(
                model_factory=lambda **kwargs: created.append(kwargs) or FakeChatModel())
        self.assertEqual(created[0]["model"], "deepseek-flash")
        self.assertEqual(created[0]["base_url"], "https://api.deepseek.com")
        self.assertEqual(created[0]["extra_body"], {"thinking": {"type": "disabled"}})


class HarnessTests(unittest.TestCase):
    def run_harness(self, responses, max_steps=5, registry=None):
        harness = main.AgentHarness(ScriptedEngine(responses), main.SYSTEM_PROMPT,
                                    main.TOOL_REGISTRY if registry is None else registry)
        with redirect_stdout(io.StringIO()) as output:
            answer = harness.run_task("Analyze report.txt and calculate total", max_steps=max_steps)
        return harness, answer, output.getvalue()

    def test_uses_real_read_calculate_and_write_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.txt"
            summary = Path(directory) / "summary.txt"
            report.write_text("Region A: 7\nRegion B: 9\n", encoding="utf-8")
            registry = {"read_file": partial(main.tool_read_file, workspace_root=directory),
                        "calculate": main.tool_calculate,
                        "write_file": partial(main.tool_write_file, workspace_root=directory)}
            harness, answer, _ = self.run_harness([
                tool_request("read_file", file_path=str(report)),
                tool_request("calculate", expression="7 + 9"),
                tool_request("write_file", file_path=str(summary), content="Total Sum: 16"),
                "<final_answer>16 saved</final_answer>",
            ], registry=registry)
            self.assertEqual(answer, "16 saved")
            self.assertEqual(summary.read_text(encoding="utf-8"), "Total Sum: 16")
            feedback = [event["content"] for event in harness.event_log if event["role"] == "user"][1:]
            self.assertEqual(len(feedback), 3)
            self.assertTrue(all(msg.startswith("<tool_output>") and msg.endswith("</tool_output>")
                                for msg in feedback))
            self.assertIn("Region A: 7", feedback[0])
            self.assertEqual(json.loads(feedback[1][len("<tool_output>"):-len("</tool_output>")])["result"]["result"], 16)

    def test_rejects_invalid_calls_before_writing(self):
        invalid = [
            '<tool_call>{broken</tool_call>',
            tool_request("unknown", file_path="anything"),
            '<tool_call>{"name":"write_file","parameters":{"file_path":"x"}}</tool_call>',
            '<tool_call>{"name":"write_file","parameters":{"file_path":"x","content":42}}</tool_call>',
            '<tool_call>{"name":"write_file","parameters":{"file_path":"x","content":"ok","extra":"bad"}}</tool_call>',
            '<tool_call>[]</tool_call>',
            '<tool_call>{"name":"write_file","parameters":[]}</tool_call>',
            tool_request("calculate", expression="1 + 1") + '<final_answer>done</final_answer>',
            tool_request("calculate", expression="1 + 1") * 2,
            'nothing actionable',
        ]
        for response in invalid:
            with self.subTest(response=response):
                calls = []
                registry = {"write_file": lambda **params: calls.append(params),
                            "calculate": lambda **params: calls.append(params)}
                harness, answer, _ = self.run_harness(
                    [response, '<final_answer>recovered</final_answer>'], registry=registry)
                self.assertEqual(answer, "recovered")
                self.assertEqual(calls, [])
                self.assertIn('"status": "error"', harness.event_log[-2]["content"])

    def test_model_error_stops_without_fabricating_success(self):
        class BrokenEngine:
            def generate(self, messages):
                raise RuntimeError("secret-bearing provider text")
        harness = main.AgentHarness(BrokenEngine(), main.SYSTEM_PROMPT, {})
        with redirect_stdout(io.StringIO()) as output:
            answer = harness.run_task("task")
        self.assertIn("model error", answer.lower())
        self.assertNotIn("secret-bearing", answer + output.getvalue())

    def test_step_limit_stops_nonfinal_responses(self):
        harness, answer, _ = self.run_harness([tool_request("calculate", expression="1 + 1")], max_steps=1)
        self.assertIn("step limit", answer)
        self.assertEqual(len(harness.event_log), 3)

    def test_does_not_print_model_thoughts(self):
        _, _, output = self.run_harness([
            "<thought>private text</thought><final_answer>done</final_answer>"])
        self.assertNotIn("private text", output)
        self.assertNotIn("<thought>", main.SYSTEM_PROMPT)


class FileToolTests(unittest.TestCase):
    def test_refuses_paths_outside_workspace_and_local_credentials(self):
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as outside:
            external = Path(outside) / "external.txt"
            external.write_text("private", encoding="utf-8")
            secret = Path(workspace) / ".env"
            secret.write_text("private", encoding="utf-8")
            for path in [external, Path(workspace) / ".." / Path(outside).name / "external.txt", secret]:
                with self.subTest(path=path):
                    self.assertEqual(main.tool_read_file(str(path), workspace_root=workspace)["status"], "error")
                    self.assertEqual(main.tool_write_file(str(path), "overwrite", workspace_root=workspace)["status"], "error")
            self.assertEqual(external.read_text(encoding="utf-8"), "private")
            self.assertEqual(secret.read_text(encoding="utf-8"), "private")

    def test_refuses_oversized_file_before_reading(self):
        with tempfile.TemporaryDirectory() as workspace:
            large = Path(workspace) / "large.txt"
            with large.open("wb") as handle:
                handle.truncate(1_048_577)
            result = main.tool_read_file("large.txt", workspace_root=workspace)
            self.assertEqual(result["status"], "error")


class DemoTests(unittest.TestCase):
    def test_demo_preserves_existing_input_and_uses_absolute_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.txt"
            report.write_text("Region A: 3\nRegion B: 4\n", encoding="utf-8")
            engine = ScriptedEngine(["<final_answer>done</final_answer>"])
            with redirect_stdout(io.StringIO()):
                result = main.run_demo(engine=engine, workspace_root=Path(directory))
            self.assertEqual(result, "done")
            self.assertEqual(report.read_text(encoding="utf-8"), "Region A: 3\nRegion B: 4\n")

    def test_demo_creates_sample_only_if_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(io.StringIO()):
                main.run_demo(engine=ScriptedEngine(["<final_answer>done</final_answer>"]),
                              workspace_root=Path(directory))
            self.assertIn("Region A: 120", (Path(directory) / "report.txt").read_text(encoding="utf-8"))

    def test_demo_requires_credentials_before_creating_files(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"AGENT_API_KEY": "replace-with-your-provider-api-key"}):
                with self.assertRaisesRegex(ValueError, "AGENT_API_KEY"):
                    main.run_demo(workspace_root=Path(directory))
            self.assertFalse((Path(directory) / "report.txt").exists())


if __name__ == "__main__":
    unittest.main()
