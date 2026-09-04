# day9_keyword_baseline

- `hit_rate@4`: 100.0% (10/10)
- Median retrieval latency: 0.046 ms
- Evaluated cases: 10

## Per-question results

| # | Result | Question | Expected source | Retrieved sources |
| ---: | --- | --- | --- | --- |
| 1 | success | Where is build_graph implemented? | src/repo_assistant/graph.py | src/repo_assistant/graph.py |
| 2 | success | Which function counts distinct overlapping terms? | src/repo_assistant/retrieval.py | src/repo_assistant/retrieval.py, README.md |
| 3 | success | Where is RepositoryChunk defined? | src/repo_assistant/loader.py | README.md, src/repo_assistant/loader.py |
| 4 | success | Which error-code-like identifier signals missing evidence? | src/repo_assistant/nodes.py | src/repo_assistant/nodes.py |
| 5 | success | How does the system rank keyword matches when several chunks tie? | src/repo_assistant/retrieval.py | README.md, src/repo_assistant/retrieval.py, src/repo_assistant/graph.py, src/repo_assistant/loader.py |
| 6 | success | What component turns repository files into line-aware chunks? | src/repo_assistant/loader.py | README.md, src/repo_assistant/loader.py, src/repo_assistant/nodes.py, src/repo_assistant/graph.py |
| 7 | success | What is the high-level architecture of the repository assistant? | README.md | README.md, src/repo_assistant/graph.py, src/repo_assistant/loader.py, src/repo_assistant/nodes.py |
| 8 | success | Which saver keeps graph checkpoints in memory? | src/repo_assistant/graph.py | src/repo_assistant/graph.py, README.md, src/repo_assistant/nodes.py, src/repo_assistant/retrieval.py |
| 9 | success | How are repository questions classified before retrieval? | src/repo_assistant/nodes.py | src/repo_assistant/nodes.py, README.md, src/repo_assistant/graph.py, src/repo_assistant/loader.py |
| 10 | success | Which payment_gateway_retry_policy handles refunds? | none | none |

## vector_only benchmark

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- `hit_rate@4`: 90.0% (9/10)
- Median retrieval latency: 4.404 ms
- Evaluated cases: 10
- Latency includes query embedding and cosine search after the one-time in-memory matrix build.
- Full per-question report: [vector_only_results.md](vector_only_results.md)

## hybrid_rrf benchmark

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Keyword model: BM25 (`rank-bm25` BM25Okapi)
- Fusion algorithm: Reciprocal Rank Fusion (RRF, `k=60`)
- Candidate depth: 20 BM25 candidates + 20 vector candidates
- `hit_rate@4`: 90.0% (9/10)
- Median retrieval latency: 3.808 ms
- Evaluated cases: 10
- Latency includes BM25 scoring, query embedding, cosine search, and reciprocal rank fusion after the one-time in-memory matrix build.
- Full per-question report: [hybrid_rrf_results.md](hybrid_rrf_results.md)

## hybrid_code_aware benchmark

- Chunking strategy: AST-based code-aware Python chunking (functions never cut in half, classes preserved as meaningful units where size permits, import-aware headers)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Keyword model: BM25 (`rank-bm25` BM25Okapi)
- Fusion algorithm: Reciprocal Rank Fusion (RRF, `k=60`)
- Candidate depth: 20 BM25 candidates + 20 vector candidates
- `hit_rate@4`: 90.0% (9/10)
- Median retrieval latency: 5.349 ms
- Evaluated cases: 10
- Precision: 9 out of 9 supported questions retrieved the exact target source file at Rank #1 with reduced candidate noise.
- Full per-question report: [hybrid_code_aware_results.md](hybrid_code_aware_results.md)


