# hybrid_rrf

- `hit_rate@4`: 90.0% (9/10)
- Median retrieval latency: 3.808 ms
- Evaluated cases: 10

## Per-question results

| # | Result | Question | Expected source | Retrieved sources |
| ---: | --- | --- | --- | --- |
| 1 | success | Where is build_graph implemented? | src/repo_assistant/graph.py | src/repo_assistant/graph.py, README.md, src/repo_assistant/loader.py, src/repo_assistant/nodes.py |
| 2 | success | Which function counts distinct overlapping terms? | src/repo_assistant/retrieval.py | src/repo_assistant/retrieval.py, README.md, src/repo_assistant/graph.py, src/repo_assistant/nodes.py |
| 3 | success | Where is RepositoryChunk defined? | src/repo_assistant/loader.py | src/repo_assistant/loader.py, README.md, src/repo_assistant/nodes.py, src/repo_assistant/graph.py |
| 4 | success | Which error-code-like identifier signals missing evidence? | src/repo_assistant/nodes.py | src/repo_assistant/nodes.py, README.md, src/repo_assistant/graph.py, src/repo_assistant/retrieval.py |
| 5 | success | How does the system rank keyword matches when several chunks tie? | src/repo_assistant/retrieval.py | src/repo_assistant/retrieval.py, README.md, src/repo_assistant/graph.py, src/repo_assistant/nodes.py |
| 6 | success | What component turns repository files into line-aware chunks? | src/repo_assistant/loader.py | src/repo_assistant/loader.py, README.md, src/repo_assistant/nodes.py, src/repo_assistant/retrieval.py |
| 7 | success | What is the high-level architecture of the repository assistant? | README.md | README.md, src/repo_assistant/loader.py, src/repo_assistant/nodes.py, src/repo_assistant/graph.py |
| 8 | success | Which saver keeps graph checkpoints in memory? | src/repo_assistant/graph.py | src/repo_assistant/graph.py, README.md, src/repo_assistant/nodes.py, src/repo_assistant/retrieval.py |
| 9 | success | How are repository questions classified before retrieval? | src/repo_assistant/nodes.py | src/repo_assistant/nodes.py, README.md, src/repo_assistant/loader.py, src/repo_assistant/graph.py |
| 10 | failure | Which payment_gateway_retry_policy handles refunds? | none | src/repo_assistant/nodes.py, README.md, src/repo_assistant/loader.py, src/repo_assistant/graph.py |
