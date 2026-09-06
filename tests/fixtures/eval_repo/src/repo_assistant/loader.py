"""Load repository files into line-aware chunks."""


class RepositoryChunk:
    """A searchable piece of a repository file."""


IGNORED_DIRECTORIES = {".git", ".venv", "node_modules", "__pycache__"}


def load_repository(repository_path):
    """Read supported repository files and create RepositoryChunk records."""

    return []
