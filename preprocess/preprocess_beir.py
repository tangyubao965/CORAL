"""Preprocess BEIR datasets for CORAL.

Expected BEIR layout:

    beir_dataset/
        corpus.jsonl          # {"_id": ..., "title": ..., "text": ...}
        queries.jsonl         # {"_id": ..., "text": ...}
        qrels/
            train.tsv         # query-id corpus-id score
            dev.tsv
            test.tsv

This script creates files used by CORAL training and inference:

    output_dir/
        corpus_processed.jsonl
        queries_processed.jsonl
        qrels_{split}.jsonl
        train.jsonl / dev.jsonl / test.jsonl
        docid2doc.json
        doc2docids.json
        docid_view_labels.json
        valid_docids.txt
        stats.json

Docid construction follows CORAL's hierarchical multi-view setting:

    [TITLE] ...     -> global view label 0
    [GLOBAL] ...    -> global view label 0
    [LOCAL-i] ...   -> local view label 1

The script uses deterministic lexical construction so it can run without any
external generator. If you already have generated global/local pseudo-queries,
you can replace the constructed docids while keeping the same output schema.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


TITLE_VIEW = 0
GLOBAL_VIEW = 0
LOCAL_VIEW = 1
IGNORE_VIEW = -100


@dataclass
class BeirDocument:
    doc_id: str
    title: str
    text: str


@dataclass
class BeirQuery:
    query_id: str
    text: str


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_corpus(path: Path) -> Dict[str, BeirDocument]:
    corpus: Dict[str, BeirDocument] = {}
    for obj in read_jsonl(path):
        doc_id = str(obj.get("_id", obj.get("id", ""))).strip()
        if not doc_id:
            continue
        corpus[doc_id] = BeirDocument(
            doc_id=doc_id,
            title=str(obj.get("title", "") or "").strip(),
            text=str(obj.get("text", "") or "").strip(),
        )
    return corpus


def load_queries(path: Path) -> Dict[str, BeirQuery]:
    queries: Dict[str, BeirQuery] = {}
    for obj in read_jsonl(path):
        query_id = str(obj.get("_id", obj.get("id", ""))).strip()
        if not query_id:
            continue
        queries[query_id] = BeirQuery(
            query_id=query_id,
            text=str(obj.get("text", "") or "").strip(),
        )
    return queries


def load_qrels(path: Path, min_relevance: int = 1) -> Dict[str, Dict[str, int]]:
    """Load BEIR qrels from TSV.

    Supports both header format:
        query-id\tcorpus-id\tscore
    and TREC-like format:
        qid\t0\tdocid\tscore
    """
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    if not path.exists():
        return qrels

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        first = next(reader, None)
        if first is None:
            return qrels

        def parse_row(row: Sequence[str]) -> Optional[Tuple[str, str, int]]:
            if len(row) < 3:
                return None
            # Header row.
            lowered = [x.lower() for x in row]
            if "query-id" in lowered or "corpus-id" in lowered or "score" in lowered:
                return None
            if len(row) >= 4:
                qid, docid, score = row[0], row[2], row[3]
            else:
                qid, docid, score = row[0], row[1], row[2]
            try:
                rel = int(float(score))
            except ValueError:
                return None
            return str(qid), str(docid), rel

        for row in [first, *reader]:
            parsed = parse_row(row)
            if parsed is None:
                continue
            qid, docid, rel = parsed
            if rel >= min_relevance:
                qrels[qid][docid] = rel
    return qrels


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_truncate(text: str, max_words: int) -> str:
    words = normalize_text(text).split()
    return " ".join(words[:max_words])


def stable_hash(text: str, length: int = 8) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def safe_identifier_text(text: str, max_words: int) -> str:
    """Make a readable but bounded textual identifier segment."""
    text = normalize_text(text)
    text = re.sub(r"[\[\]\|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return word_truncate(text, max_words=max_words)


def split_local_chunks(text: str, max_chunks: int, chunk_words: int) -> List[str]:
    words = normalize_text(text).split()
    if not words:
        return []
    chunks: List[str] = []
    for i in range(0, len(words), chunk_words):
        if len(chunks) >= max_chunks:
            break
        chunk = " ".join(words[i : i + chunk_words])
        if chunk:
            chunks.append(chunk)
    return chunks


def make_unique_docid(candidate: str, used: set[str], doc_id: str) -> str:
    """Ensure docid uniqueness without losing readability."""
    candidate = normalize_text(candidate)
    if candidate not in used:
        used.add(candidate)
        return candidate
    suffix = stable_hash(doc_id + "::" + candidate)
    unique = f"{candidate} [DOC-{suffix}]"
    counter = 1
    while unique in used:
        counter += 1
        unique = f"{candidate} [DOC-{suffix}-{counter}]"
    used.add(unique)
    return unique


def construct_docids(
    doc: BeirDocument,
    used_docids: set[str],
    max_title_words: int = 16,
    max_global_words: int = 48,
    max_local_chunks: int = 3,
    local_chunk_words: int = 64,
    include_title_docid: bool = True,
    include_global_docid: bool = True,
    include_local_docids: bool = True,
) -> List[Tuple[str, int, str]]:
    """Construct CORAL-style multi-view docids.

    Returns a list of tuples: (docid, view_label, view_name).
    """
    results: List[Tuple[str, int, str]] = []
    title = safe_identifier_text(doc.title, max_title_words)
    body = safe_identifier_text(doc.text, max_global_words)
    fallback = f"document {stable_hash(doc.doc_id)}"

    if include_title_docid:
        title_segment = title or body or fallback
        docid = make_unique_docid(f"[TITLE] {title_segment}", used_docids, doc.doc_id)
        results.append((docid, TITLE_VIEW, "title"))

    if include_global_docid:
        if title and body:
            global_segment = f"{title} | {body}"
        else:
            global_segment = title or body or fallback
        docid = make_unique_docid(f"[GLOBAL] {global_segment}", used_docids, doc.doc_id)
        results.append((docid, GLOBAL_VIEW, "global"))

    if include_local_docids:
        chunks = split_local_chunks(doc.text, max_chunks=max_local_chunks, chunk_words=local_chunk_words)
        if not chunks and body:
            chunks = [body]
        for idx, chunk in enumerate(chunks, start=1):
            segment = safe_identifier_text(chunk, max_words=max_global_words)
            if title:
                segment = f"{title} | {segment}"
            docid = make_unique_docid(f"[LOCAL-{idx}] {segment}", used_docids, f"{doc.doc_id}-{idx}")
            results.append((docid, LOCAL_VIEW, "local"))

    return results


def build_docid_maps(
    corpus: Dict[str, BeirDocument],
    args: argparse.Namespace,
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, int], Dict[str, str]]:
    docid2doc: Dict[str, str] = {}
    doc2docids: Dict[str, List[str]] = {}
    docid_view_labels: Dict[str, int] = {}
    docid_view_names: Dict[str, str] = {}
    used_docids: set[str] = set()

    for doc_id, doc in corpus.items():
        entries = construct_docids(
            doc,
            used_docids=used_docids,
            max_title_words=args.max_title_words,
            max_global_words=args.max_global_words,
            max_local_chunks=args.max_local_docids,
            local_chunk_words=args.local_chunk_words,
            include_title_docid=not args.no_title_docid,
            include_global_docid=not args.no_global_docid,
            include_local_docids=not args.no_local_docids,
        )
        doc2docids[doc_id] = []
        for docid, view_label, view_name in entries:
            docid2doc[docid] = doc_id
            doc2docids[doc_id].append(docid)
            docid_view_labels[docid] = view_label
            docid_view_names[docid] = view_name
    return docid2doc, doc2docids, docid_view_labels, docid_view_names


def make_training_rows(
    queries: Dict[str, BeirQuery],
    qrels: Dict[str, Dict[str, int]],
    doc2docids: Dict[str, List[str]],
    docid_view_labels: Dict[str, int],
    docid_view_names: Dict[str, str],
    max_rows_per_query: Optional[int] = None,
    seed: int = 42,
) -> Iterator[dict]:
    rng = random.Random(seed)
    for qid, doc_scores in qrels.items():
        if qid not in queries:
            continue
        positives: List[Tuple[str, str]] = []
        for doc_id in doc_scores:
            for docid in doc2docids.get(doc_id, []):
                positives.append((doc_id, docid))
        if max_rows_per_query is not None and len(positives) > max_rows_per_query:
            positives = rng.sample(positives, max_rows_per_query)
        for doc_id, docid in positives:
            yield {
                "query_id": qid,
                "query": queries[qid].text,
                "doc_id": doc_id,
                "target_docid": docid,
                "view_label": docid_view_labels.get(docid, IGNORE_VIEW),
                "view_name": docid_view_names.get(docid, "unknown"),
                "relevance": doc_scores.get(doc_id, 1),
            }


def make_eval_rows(
    queries: Dict[str, BeirQuery],
    qrels: Dict[str, Dict[str, int]],
) -> Iterator[dict]:
    for qid, query in queries.items():
        positive_docs = qrels.get(qid, {})
        yield {
            "query_id": qid,
            "query": query.text,
            "positive_doc_ids": list(positive_docs.keys()),
            "relevance": positive_docs,
        }


def find_qrel_files(beir_dir: Path, requested_splits: Sequence[str]) -> Dict[str, Path]:
    qrels_dir = beir_dir / "qrels"
    files: Dict[str, Path] = {}
    for split in requested_splits:
        candidates = [
            qrels_dir / f"{split}.tsv",
            qrels_dir / f"{split}.txt",
            beir_dir / f"qrels.{split}.tsv",
            beir_dir / f"{split}.tsv",
        ]
        for candidate in candidates:
            if candidate.exists():
                files[split] = candidate
                break
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess BEIR data for CORAL.")
    parser.add_argument("--beir_dir", type=str, required=True, help="Path to one BEIR dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for processed files.")
    parser.add_argument("--splits", type=str, default="train,dev,test", help="Comma-separated qrel splits to process.")
    parser.add_argument("--min_relevance", type=int, default=1)
    parser.add_argument("--max_title_words", type=int, default=16)
    parser.add_argument("--max_global_words", type=int, default=48)
    parser.add_argument("--max_local_docids", type=int, default=3)
    parser.add_argument("--local_chunk_words", type=int, default=64)
    parser.add_argument("--max_rows_per_query", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_title_docid", action="store_true")
    parser.add_argument("--no_global_docid", action="store_true")
    parser.add_argument("--no_local_docids", action="store_true")
    args = parser.parse_args()

    beir_dir = Path(args.beir_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = beir_dir / "corpus.jsonl"
    queries_path = beir_dir / "queries.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus file: {corpus_path}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")

    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    docid2doc, doc2docids, docid_view_labels, docid_view_names = build_docid_maps(corpus, args)

    # Save corpus and query files in a stable project-friendly schema.
    write_jsonl(
        output_dir / "corpus_processed.jsonl",
        (
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "text": doc.text,
                "docids": doc2docids.get(doc.doc_id, []),
            }
            for doc in corpus.values()
        ),
    )
    write_jsonl(
        output_dir / "queries_processed.jsonl",
        ({"query_id": q.query_id, "query": q.text} for q in queries.values()),
    )

    with (output_dir / "docid2doc.json").open("w", encoding="utf-8") as f:
        json.dump(docid2doc, f, ensure_ascii=False, indent=2)
    with (output_dir / "doc2docids.json").open("w", encoding="utf-8") as f:
        json.dump(doc2docids, f, ensure_ascii=False, indent=2)
    with (output_dir / "docid_view_labels.json").open("w", encoding="utf-8") as f:
        json.dump(docid_view_labels, f, ensure_ascii=False, indent=2)
    with (output_dir / "docid_view_names.json").open("w", encoding="utf-8") as f:
        json.dump(docid_view_names, f, ensure_ascii=False, indent=2)
    with (output_dir / "valid_docids.txt").open("w", encoding="utf-8") as f:
        for docid in docid2doc:
            f.write(docid + "\n")

    split_names = [x.strip() for x in args.splits.split(",") if x.strip()]
    qrel_files = find_qrel_files(beir_dir, split_names)
    split_stats = {}

    for split in split_names:
        qrel_path = qrel_files.get(split)
        if qrel_path is None:
            split_stats[split] = {"found": False, "num_qrels": 0, "num_rows": 0}
            continue

        qrels = load_qrels(qrel_path, min_relevance=args.min_relevance)
        write_jsonl(output_dir / f"qrels_{split}.jsonl", make_eval_rows(queries, qrels))

        rows_path = output_dir / f"{split}.jsonl"
        num_rows = write_jsonl(
            rows_path,
            make_training_rows(
                queries=queries,
                qrels=qrels,
                doc2docids=doc2docids,
                docid_view_labels=docid_view_labels,
                docid_view_names=docid_view_names,
                max_rows_per_query=args.max_rows_per_query,
                seed=args.seed,
            ),
        )
        split_stats[split] = {
            "found": True,
            "qrel_file": str(qrel_path),
            "num_queries_with_qrels": len(qrels),
            "num_positive_pairs": sum(len(v) for v in qrels.values()),
            "num_training_rows": num_rows,
        }

    stats = {
        "num_documents": len(corpus),
        "num_queries": len(queries),
        "num_docids": len(docid2doc),
        "avg_docids_per_doc": len(docid2doc) / max(len(corpus), 1),
        "splits": split_stats,
        "view_label_schema": {"global_or_title": 0, "local": 1, "ignore": -100},
        "docid_schema": {
            "title": "[TITLE] <title>",
            "global": "[GLOBAL] <title> | <document prefix>",
            "local": "[LOCAL-i] <title> | <local chunk>",
        },
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
