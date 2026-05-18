import json
from pathlib import Path

def split_into_passages(text, max_tokens=128, stride=64):
    tokens = text.split()
    passages = []
    for i in range(0, len(tokens), stride):
        passages.append(" ".join(tokens[i:i+max_tokens]))
        if i + max_tokens >= len(tokens):
            break
    return passages

def construct_local_view_input(docs_path, output_path, max_passages=4, max_tokens=128, stride=64):
    output = []
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc["doc_id"]
            text = doc["text"]
            passages = split_into_passages(text, max_tokens=max_tokens, stride=stride)
            for i, passage in enumerate(passages[:max_passages]):
                output.append({
                    "doc_id": doc_id,
                    "passage_id": f"{doc_id}_passage_{i}",
                    "passage": passage,
                    "view": "local"
                })
    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in output:
            out_f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_passages", type=int, default=4)
    args = parser.parse_args()

    construct_local_view_input(
        docs_path=args.docs_path,
        output_path=args.output_path,
        max_passages=args.max_passages
    )
