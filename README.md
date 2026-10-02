# Week 1: Agent Runtime Harness

This demo keeps its own explicit agent/tool loop and uses LangChain's `ChatOpenAI` only as the chat-model adapter. The model reads `report.txt`, calls the calculator, writes `summary.txt`, and returns a final answer. Model output is validated before any tool executes.

## Setup

From the project root (`AI Agent Demo`), install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run `uv sync`. The root project requires Python 3.14 or newer; uv uses its project environment automatically. `Week 1` is also a separate uv project, so run the commands below **from the project root** unless stated otherwise.

Set `AGENT_API_KEY` in the ignored project-root `.env` to your **own provider API key** (not a Copilot subscription token). Keep the key private; do not commit `.env` or paste its contents into logs. If `.env` is missing, create it with `AGENT_API_KEY=replace-with-your-provider-api-key`, then replace that placeholder locally. The default provider is DeepSeek using `AGENT_MODEL=deepseek-flash` and `AGENT_BASE_URL=https://api.deepseek.com` (these two lines are optional). DeepSeek thinking mode is disabled for its official endpoint because this harness expects a single XML action in the response text on each turn. Environment variables take precedence over `.env`. For another OpenAI-compatible provider, set all three variables to its key, supported model ID, and base URL.

GitHub Models' inference API was [retired on July 30, 2026](https://docs.github.com/en/github-models/quickstart); it cannot be used as an alternate endpoint. GitHub Copilot itself is a separate service. DeepSeek model/endpoint details: [official API docs](https://api-docs.deepseek.com/).

## Complete end-to-end run

1. Run the free, offline tests first from the project root: `uv run python -m unittest discover -s tests -v`. They inject fake models, need no API key, and make no provider requests.
2. Check the root `report.txt`. If it is absent, the demo creates a sample report with region values 120, 280, and 450. If it already exists, the demo **does not overwrite it** and calculates from its actual contents.
3. With your API key configured, run `uv run './Week 1/main.py'` from the project root (or `cd 'Week 1'` and then `uv run main.py`). This is a **live provider call** that may incur charges; it uses up to five model turns.
4. Look for `read_file` → `calculate` → `write_file` → `final_answer` in the console, then open the **root** `summary.txt`. For the sample report, the computed total should be **850**. The model chooses the summary's formatting (for example, it may write just `850`), so check the number rather than expecting an exact template. A successful run typically takes four turns; an error or step-limit warning is not success.

The validated `write_file` action **can overwrite an existing `summary.txt`**; save a copy first if you need it. Console output includes parsed model actions and file content, so do not share logs containing sensitive workspace data. Missing credentials or a model failure do not produce a fabricated success response.

This educational example exposes local read/write tools to the model. Paths are restricted to the project workspace, `.env`/`.venv`/`.git` are blocked, and reads over 1 MiB are rejected. This is not an operating-system sandbox: trusted inputs and a disposable workspace are recommended, since other workspace files can still be modified.
