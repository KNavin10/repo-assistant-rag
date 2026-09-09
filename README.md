# repo-assistant-rag

A retrieval-augmented generation (RAG) assistant for codebase exploration, architectural analysis, and semantic search with precise line-level citations and Model Context Protocol (MCP) server support.

## Retrieval Benchmarks

Measured on the 10-question evaluation dataset (`evals/retrieval_cases.yaml`) against the fixture repository (`tests/fixtures/eval_repo`):

| Retriever | Hit rate @ 4 | Median latency |
| :--- | :---: | :---: |
| Day 9 keyword baseline | 100.0% | 0.046 ms |
| Vector only | 90.0% | 4.404 ms |
| Hybrid RRF | 90.0% | 3.808 ms |
| Hybrid + code-aware chunks | 90.0% | 4.994 ms |

*Detailed benchmark reports and per-question breakdowns are available in [evals/results.md](evals/results.md), [evals/vector_only_results.md](evals/vector_only_results.md), [evals/hybrid_rrf_results.md](evals/hybrid_rrf_results.md), and [evals/hybrid_code_aware_results.md](evals/hybrid_code_aware_results.md).*

## Features

- **Code-Aware Chunking**: Uses Python's `ast` module to preserve atomic functions and classes, with import-scoped headers and non-Python fallback chunking.
- **Hybrid RRF Retrieval**: Combines BM25 lexical search with dense vector embeddings (`all-MiniLM-L6-v2`) using Reciprocal Rank Fusion.
- **Strict Citation Validation**: Enforces verifiable `file_path:start_line-end_line` citations matching repository files on disk and retrieved chunks.
- **Read-Only Security Boundary**: Rejects path traversal, sensitive and ignored files, oversized scans, and unbounded MCP inputs.
- **MCP Server**: Provides standard Model Context Protocol v2 tools (`search_code` and `explain_file`) on stdio.

The MCP server confines repository selection to the directory from which it is
started. Tests and embedded callers can provide a narrower `allowed_root` to
`create_mcp_server`.

## External-Service Consent

The CLI will not send retrieved repository content to Groq unless the request includes
`--allow-external-model`. If LangSmith tracing is enabled, repositories outside
`tests/fixtures` also require the separate `--allow-non-fixture-tracing` flag.

## Running Tests and Evaluations

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .day10-test-temp
.\.venv\Scripts\python.exe -m repo_assistant.evaluation
.\.venv\Scripts\python.exe -m repo_assistant.security_evaluation
```
