# Repository Assistant

This repository implements a repository assistant with a small, provider-free architecture.
The loader reads source files into line-aware RepositoryChunk records. Keyword retrieval ranks
chunks by distinct term overlap, and the LangGraph workflow validates, loads, routes, retrieves,
and generates cited answers. Graph checkpoints use InMemorySaver.
