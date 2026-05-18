import argparse
import json
from tqdm import tqdm

def load_query_docid_mapping(query_file):

    with open(query_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_docid_text(docid_file):

    with open(docid_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_labeled_pairs(query_file, docid_file, output_file):
    query_docid_data = load_query_docid_mapping(query_file)
    docid_text_data = load_docid_text(docid_file)

    pairs = []

    for qid, qdata in tqdm(query_docid_data.items(), desc="Building labeled pairs"):
        query = qdata['query']
        doc_id = qdata['doc_id']
        doc_text = docid_text_data.get(doc_id, None)

        if doc_text is not None:
            pairs.append({
                'query_id': qid,
                'query': query,
                'doc_id': doc_id,
                'doc_text': doc_text
            })

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in pairs:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_file', type=str, required=True, help="Path to the labeled queries JSON file")
    parser.add_argument('--docid_file', type=str, required=True, help="Path to the document ID text mapping JSON file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the output labeled pairs")

    args = parser.parse_args()

    build_labeled_pairs(args.query_file, args.docid_file, args.output_file)
