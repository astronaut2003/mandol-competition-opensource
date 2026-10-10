"""Run the canonical Qdrant-only AML deployment from environment."""

from mandol_competition.textual.core.deployment import serve

from .config import Settings


def main():
    serve(Settings.from_env())


if __name__ == "__main__":
    main()
