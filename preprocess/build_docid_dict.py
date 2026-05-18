import json
from collections import defaultdict
from tqdm import tqdm

def build_docid_mapping(docid_file, output_file):

    docid_to_doc = dict()
    with open(docid_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Building docid mapping"):
            data = json.loads(line)
            docid = data["docid"]
            doc_text = data["text"]
            docid_to_doc[docid] = doc_text

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(docid_to_doc, f, ensure_ascii=False, indent=2)

def build_reverse_mapping(docid_to_doc_file, output_file):
       with open(docid_to_doc_file, 'r', encoding='utf-8') as f:
        docid_to_doc = json.load(f)

    doc_to_docids = defaultdict(list)
    for docid, text in docid_to_doc.items():
        doc_to_docids[text].append(docid)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(doc_to_docids, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to docid input file")
    parser.add_argument("--output", type=str, required=True, help="Path to save docid-to-doc mapping")
    parser.add_argument("--reverse_output", type=str, default=None, help="Path to save doc-to-docids mapping")
    args = parser.parse_args()

    build_docid_mapping(args.input, args.output)
    if args.reverse_output:
        build_reverse_mapping(args.output, args.reverse_output)
