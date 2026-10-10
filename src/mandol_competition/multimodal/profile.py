"""Freeze a retrieval recipe for a fresh canonical database (no inference)."""

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from mandol_substrate.state import RetrievalProfile
from mandol_substrate.text import MandolTextProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-revision", default="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    parser.add_argument("--splade-model", default="naver/splade-v3")
    parser.add_argument("--splade-revision", default="fdfeceb91d7b9de7985b38addd3ba9f53a59a355")
    parser.add_argument("--query-prompt", default="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ")
    parser.add_argument("--jieba", action="store_true")
    parser.add_argument("--no-reranker", action="store_true")
    args = parser.parse_args()
    processor = MandolTextProcessor(use_jieba=args.jieba)
    profile = RetrievalProfile(args.embedding_model, args.embedding_revision, args.embedding_dimension,
        args.embedding_revision, args.splade_model, args.splade_revision, processor.fingerprint,
        methods=("bm25", "cosine", "splade"), options={
            "text_mode": "regex", "jieba": args.jieba, "model_device": args.device,
            "dense_max_length": 4096, "dense_query_prompt": args.query_prompt,
            "dense_microbatch": 1, "dense_dtype": "float32", "splade_microbatch": 1,
            "splade_max_length": 512, "splade_dtype": "float32", "candidate_k": 100,
            "rerank_method": None if args.no_reranker else "baai",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "reranker_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "reranker_max_length": 4096, "reranker_microbatch": 1,
            "inference_recipe": "multimodal-local-fp32-v1",
            "runtime": json.loads(Path(__file__).with_name("profiles").joinpath("frozen-fp32-v1.json").read_text())["options"]["runtime"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(asdict(profile), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(profile.hash)


if __name__ == "__main__":
    main()
