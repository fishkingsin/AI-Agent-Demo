# Week 2 LLM API Integration Design

Date: 2026-10-03
Status: Design approved; implementation not started.

## Goal and scope

Replace the hard-coded answer and reflection in `Week 2/week2_hybrid_rag.py` with API-backed model calls, using the same configurable OpenAI-compatible model integration as Week 1. Keep BM25, local TF-IDF cosine retrieval, and Reciprocal Rank Fusion local and deterministic. The scope is generation and review after retrieval—not an autonomous agent or a replacement for the retrievers.

## Recommended architecture

Extract Week 1's `LangChainLLMEngine` into a small shared root-level module (for example, `common/llm_engine.py`). Keep its simple `generate(messages) -> str` interface and use it from both Week 1 and Week 2. Preserve a compatible import for Week 1's existing entry point/tests. Make repository-root resolution explicit so the shared module and root `.env` work in both documented Week 1 launch modes and when running Week 2. Keep dependencies available in both project configurations as necessary.

Alternatives considered:

1. Import the adapter directly from `Week 1/main.py`: minimal duplication, but couples Week 2 to a script path and unrelated harness code.
2. **Extract the shared module (recommended):** a small refactor, with one provider/configuration implementation for both demos.
3. Copy the adapter into Week 2: self-contained, but allows configuration and behavior to drift.

The adapter uses Week 1's `AGENT_API_KEY`, `AGENT_MODEL`, and `AGENT_BASE_URL`, and loads the existing project-root `.env` without overriding explicit environment values. No new credential scheme is introduced.

## Workflow and data flow

1. Keep the existing deterministic plan and local hybrid retrieval.
2. Package the query and retrieved documents, including document IDs, as context for one answer-generation call. Ask for a concise answer grounded only in the supplied context, with citations using those IDs. Do not include retrieved content as trusted instructions.
3. Send the generated answer, query, and same retrieved context to a second call for reflection. Request a small structured JSON verdict covering whether claims are supported, whether citations are valid, unsupported claims/issues, and an overall status.
4. Validate the reflection structure and types; independently ensure cited document IDs belong to the retrieved set. Malformed output or invalid citations must not produce an approved verdict.
5. Return the existing workflow result fields (`query`, `plan`, `context_xml`, `output`, `reflection`) while representing failures/inconclusive checks honestly. An LLM reflection is a review signal, not a factual guarantee.

If retrieval returns no documents, skip both model calls and return an explicit insufficient-context answer with an inconclusive reflection. If answer generation fails, do not fabricate an answer. If reflection fails after an answer was generated, preserve the answer but report reflection as unavailable/inconclusive; never claim approval.

## Testing and configuration

Use injected fake models/adapters for offline tests; normal tests require neither a key nor network access. Cover the two-call sequence and prompts, retrieved IDs and citation validation, malformed reflection, empty retrieval (zero calls), and answer/reflection provider failures. Retain Week 1's adapter and harness tests after extraction, and test the documented launch paths/configuration. Keep any live-provider smoke test optional and explicitly credential-gated.

The existing project-root `.env` should remain ignored and must never be read into logs or committed. Do not add credentials to fixtures or documentation. Update dependency metadata only where needed to ensure both demo entry points can import and run the shared adapter.

## Out of scope

- API embeddings or a vector database; the Week 2 vector retriever remains its current local term-frequency cosine implementation.
- Changes to BM25, vector scoring, RRF ranking, or retrieval query planning.
- Model-controlled tools or retrieval decisions, automatic answer regeneration, and claims that reflection guarantees factuality.
