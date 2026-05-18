import json
import argparse
from tqdm import tqdm

def build_doc_docid_pairs(docid_files, output_file):
    '''
    Given multiple docid files (e.g., global, local, title), 
    combine them to form document-docid training pairs.
    '''
    docid_map = {}  # doc_id -> list of (docid, view)

    for file_path in docid_files:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                doc_id = data["doc_id"]
                docid = data["docid"]
                view = data["view"]
                if doc_id not in docid_map:
                    docid_map[doc_id] = []
                docid_map[doc_id].append((docid, view))

    # Write pairs to output
    with open(output_file, 'w') as fout:
        for doc_id, docids in docid_map.items():
            for docid, view in docids:
                fout.write(json.dumps({
                    "doc_id": doc_id,
                    "docid": docid,
                    "view": view
                }) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docid_files", nargs='+', required=True,
                        help="List of input docid files (global, local, title)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output path for doc-docid training pairs")
    args = parser.parse_args()

    build_doc_docid_pairs(args.docid_files, args.output_file)