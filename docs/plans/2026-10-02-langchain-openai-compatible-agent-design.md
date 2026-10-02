# LangChain model adapter for the Week 1 agent harness

Date: 2026-10-02
Status: Design approved; implemented and tested offline.

## Goal and scope

Replace `SimulatedLLMEngine` in `Week 1/main.py` with a real, LangChain-backed model connection. Default to DeepSeek while accepting other OpenAI-compatible endpoints via configuration. Keep the Week 1 lesson—the custom, visible `AgentHarness` tool loop and append-only event log—intact. GitHub Models' inference API was retired July 30, 2026; a GitHub Copilot subscription is not an API credential for this adapter.

## Approach and alternatives

Recommended: keep `AgentHarness`, and introduce a `LangChainLLMEngine` implementing `generate(messages) -> str` using `langchain-openai`'s `ChatOpenAI`. This minimizes churn and leaves step control, tool execution, and logging in the harness. Replacing the harness with a LangChain agent would provide built-in orchestration, but would obscure the custom runtime the exercise demonstrates. A hybrid using model-native tool calling mapped into the old XML protocol adds complexity without a Week 1 benefit; defer it.

## Components and configuration

- `LangChainLLMEngine` converts system/user/assistant event-log entries to LangChain chat messages, invokes the configured chat model once per step, and returns text for the existing harness to parse. Reject an unsupported, non-text model response clearly.
- Configure `AGENT_API_KEY`, `AGENT_MODEL` (default `deepseek-flash`), and `AGENT_BASE_URL` (default `https://api.deepseek.com`) through the environment. Load a local `.env` for development; never embed a real credential in code, fixtures, or documentation. Changing the base URL, model, and key selects another currently available OpenAI-compatible provider. DeepSeek model names and endpoint were verified against its current API reference.
- Add a small dependency manifest containing `langchain-openai` and `python-dotenv`; document setup and how to run the sample. The project-root `.env` starts with placeholders, may be filled in locally, and must be Git-ignored.
- Keep the existing `read_file`, `calculate`, and `write_file` tools. LangChain supplies model access, not tool execution; avoid adding a second tool loop.

## Data flow and protocol

1. Append the user task to the event log and prepend `SYSTEM_PROMPT` on each model request.
2. Receive exactly one XML-wrapped `<tool_call>` or `<final_answer>` per step. The prompt will request only a concise action summary, not private chain-of-thought or `<thought>` tags; do not print model-generated private reasoning.
3. Validate a tool call's JSON shape, registered name, and required/allowed argument types before invoking a tool. Append a structured `<tool_output>` containing the real result or a descriptive validation error, aligning the prompt and harness. Treat tool output as untrusted data rather than instructions.
4. Continue until a final answer or the existing step limit. A response with both an action and final answer, multiple actions, or neither receives explicit feedback or terminates safely without performing an ambiguous action.
5. The demo should use actual `report.txt` contents rather than hard-coded numbers or success messages. Avoid silently overwriting existing user data for the input example. File tools must stay inside the selected workspace and refuse local credentials; read size is capped at 1 MiB.

## Failure behavior and tests

Missing credentials fail before any API request with setup guidance. Network, authentication, rate-limit, and malformed response failures stop clearly instead of fabricating success. An unknown tool, invalid JSON, unsupported arguments, a tool-level error, and the step limit have predictable logged outcomes. For any write request, execute only an explicitly validated tool call.

Inject a fake engine to test the full read → calculate → write → final-answer sequence offline, including modified input numbers, invalid calls, API failure propagation, and the step limit. Keep an optional credential-gated smoke test for DeepSeek or another active provider; do not require network access for the normal test suite. Validate that logs contain actual tool outputs and never reveal credentials or model-generated private reasoning.

## Repository status

The workspace is not a Git repository, so this design cannot currently be committed. Do not initialize a repository without the user's approval.
