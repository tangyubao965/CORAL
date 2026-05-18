"""Visual analysis for query and docid representations.

Input JSONL example:
{"id": "q1", "type": "query", "method": "CORAL", "label": "query", "vector": [...]}
{"id": "docid1", "type": "docid", "method": "CORAL", "label": "global", "vector": [...]}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis.io_utils import read_json, read_jsonl, write_csv


def load_representation_rows(path: str):
    if path.endswith(".jsonl"):
        return read_jsonl(path)
    obj = read_json(path)
    if isinstance(obj, dict) and "representations" in obj:
        return obj["representations"]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported representation file: {path}")


def project(vectors, method: str, seed: int):
    x = np.asarray(vectors, dtype=np.float32)
    if method == "pca":
        x = x - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        return x @ vt[:2].T

    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, max(2, (len(x) - 1) // 3))
        return TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca").fit_transform(x)

    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repr", required=True, help="Representation JSON or JSONL")
    parser.add_argument("--method", default="tsne", choices=["tsne", "pca"])
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_png", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_representation_rows(args.repr)
    coords = project([row["vector"] for row in rows], method=args.method, seed=args.seed)

    out_rows = []
    for row, xy in zip(rows, coords):
        out_rows.append({
            "id": row.get("id", ""),
            "type": row.get("type", ""),
            "method": row.get("method", ""),
            "label": row.get("label", ""),
            "x": float(xy[0]),
            "y": float(xy[1]),
        })
    write_csv(out_rows, args.out_csv)

    if args.out_png:
        import matplotlib.pyplot as plt
        Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 5))
        labels = sorted(set(row["label"] for row in out_rows))
        for label in labels:
            xs = [row["x"] for row in out_rows if row["label"] == label]
            ys = [row["y"] for row in out_rows if row["label"] == label]
            plt.scatter(xs, ys, s=16, alpha=0.75, label=label)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
