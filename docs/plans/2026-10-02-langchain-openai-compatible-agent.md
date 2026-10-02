# LangChain OpenAI-Compatible Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the hard-coded simulator with an actual configurable LangChain chat model while retaining the custom tool-loop demonstration.

**Architecture:** `LangChainLLMEngine.generate` adapts logged messages to LangChain roles, invokes `ChatOpenAI` once, and returns text. `AgentHarness` validates a single XML action and executes registered tools, feeding back real results until final answer or step limit. Offline tests use injected fake engines/models.

**Tech Stack:** Python 3.14, `langchain-openai`, `python-dotenv`, stdlib `unittest`.

---

### Task 1: Model adapter and configuration

**Files:** Modify `Week 1/main.py`; create `tests/test_main.py`, `requirements.txt`.

1. Write tests using importlib to load `Week 1/main.py` for `LangChainLLMEngine`: `generate` maps system/user/assistant roles to LangChain messages, invokes an injected fake chat model exactly once, returns text, and rejects non-text responses. Test missing/placeholder API key before any model construction and configurable base URL/model passed to the chat constructor.
2. Run `python3 -m unittest discover -s tests -v`; expect failures because `LangChainLLMEngine` does not exist.
3. Add the smallest adapter and environment loader (`load_dotenv` from project-root `.env`, without overwriting real environment values). Declare `langchain-openai` and `python-dotenv` in `requirements.txt`; install into an isolated environment before tests.
4. Rerun the entire offline test command; expect all tests passing.

### Task 2: Harness protocol and safe dispatch

**Files:** Modify `Week 1/main.py`; extend `tests/test_main.py`.

1. Test the read → calculate → write → final sequence using a fake engine and temporary file contents different from 120/280/450. Assert each tool result is present in `<tool_output>` and final answer returned.
2. Run tests to watch the new assertions fail on `Tool_Output (...)` and hard-coded behavior.
3. Implement aligned output, unambiguous single-action extraction, schema argument checking, error feedback for invalid JSON, unknown tools, wrong types, extra/missing fields, and safe handling of tool exceptions. Remove `<thought>` instructions/examples and log printing, replacing with optional brief action summaries only.
4. Add failing tests for malformed/multiple actions, both action and final answer, no action, tool failure, model/API failure, and max steps; implement one case at a time until green. `write_file` must not run for an invalid action.
5. Run the full test command after each increment.

### Task 3: Entrypoint, docs, verification

**Files:** Modify `Week 1/main.py`, `.gitignore`; create `README.md`; extend `tests/test_main.py`.

1. Test that running from any working directory resolves `report.txt` in the workspace, does not overwrite an existing report, and missing credentials produce a clear error rather than success. Verify that the input numbers are read dynamically.
2. Run tests; confirm expected failure from the old entrypoint.
3. Update entrypoint to load project-root configuration, create an example report only when absent, use workspace-root paths and instantiate the adapter. Document venv/dependency setup, `.env` credential configuration, current DeepSeek defaults, offline tests, and lack of automatic Copilot subscription credential access. GitHub Models was retired July 30, 2026; use another active OpenAI-compatible endpoint instead. Ignore virtual environments and caches.
4. Run `.venv/bin/python -m unittest discover -s tests -v`, `.venv/bin/python -m compileall -q 'Week 1' tests`, and editor diagnostics. Live calls require a configured key, may incur charges, and are not required for offline verification. No Git commit or worktree: this workspace has no Git repository.
