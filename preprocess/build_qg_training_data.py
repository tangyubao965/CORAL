import json
import os
from tqdm import tqdm
import argparse
import hashlib

def hash_text(text):
    return hashlib.md5(text.encode()).hexdigest()

def split_into_passages(text, num_passages):
    sentences = text.strip().split(". ")
    total = len(sentences)
    avg_len = max(1, total // num_passages)
    return [" ".join(sentences[i:i+avg_len]) for i in range(0, total, avg_len)][:num_passages]

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_json, 'r') as f:
        docs = json.load(f)

    global_pairs = []
    local_pairs = []

    for doc in tqdm(docs, desc="Processing"):
        doc_id = doc["doc_id"]
        content = doc["content"]
        real_queries = doc.get("real_queries", [])

        if not real_queries:
            continue

        global_query = real_queries[0]
        global_pairs.append({
            "doc_input": content,
            "query": global_query,
            "view": "global",
            "docid": f"{doc_id}_g"
        })

        passages = split_into_passages(content, args.num_passages)
        for idx, passage in enumerate(passages):
            if idx + 1 >= len(real_queries):
                break
            local_query = real_queries[idx + 1]
            local_pairs.append({
                "doc_input": passage,
                "query": local_query,
                "view": "local",
                "docid": f"{doc_id}_l{idx}"
            })

    all_pairs = global_pairs + local_pairs
    with open(os.path.join(args.output_dir, "qg_training_pairs.json"), 'w') as f:
        json.dump(all_pairs, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_passages", type=int, default=3)
    args = parser.parse_args()
    main(args)
