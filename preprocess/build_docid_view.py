
import os
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer


def chunk_text(text, chunk_size=128, stride=64):
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + chunk_size]
        if len(chunk) < 10:  # skip short chunks
            continue
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(tokens):
            break
    return chunks


def build_views(doc_path, output_path, max_passages=3):
    with open(doc_path, 'r') as f:
        docs = [json.loads(line) for line in f]

    os.makedirs(output_path, exist_ok=True)

    global_view_path = os.path.join(output_path, "docid_global.jsonl")
    local_view_path = os.path.join(output_path, "docid_local.jsonl")
    title_view_path = os.path.join(output_path, "docid_title.jsonl")

    with open(global_view_path, 'w') as fg,          open(local_view_path, 'w') as fl,          open(title_view_path, 'w') as ft:

        for doc in tqdm(docs, desc="Building views"):
            docid = doc['docid']
            title = doc.get('title', '')
            text = doc.get('text', '')

            fg.write(json.dumps({
                "docid": docid,
                "view": "global",
                "text": text
            }) + '\n')

            if title.strip():
                ft.write(json.dumps({
                    "docid": docid,
                    "view": "title",
                    "text": title
                }) + '\n')

            passages = chunk_text(text)
            for i, passage in enumerate(passages[:max_passages]):
                fl.write(json.dumps({
                    "docid": f"{docid}_p{i}",
                    "view": "local",
                    "parent_docid": docid,
                    "text": passage
                }) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_path", type=str, required=True, help="Path to the input document JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for view docids")
    parser.add_argument("--max_passages", type=int, default=3, help="Max number of local passages per document")
    args = parser.parse_args()

    build_views(args.doc_path, args.output_path, args.max_passages)
