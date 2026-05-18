
import argparse
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import torch
import os

def load_documents(doc_path):
    with open(doc_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_queries(model, tokenizer, documents, view_type, max_input_length=512, max_output_length=32, batch_size=8, device='cuda'):
    model.to(device)
    model.eval()
    results = []
    with torch.no_grad():
        for doc in tqdm(documents, desc=f"Generating {view_type} queries"):
            doc_id = doc['doc_id']
            if view_type == 'global':
                inputs = tokenizer(doc['full_text'], return_tensors='pt', truncation=True, max_length=max_input_length).to(device)
                output = model.generate(**inputs, max_length=max_output_length, num_return_sequences=1)
                query = tokenizer.decode(output[0], skip_special_tokens=True)
                results.append({'doc_id': doc_id, 'query': query, 'view': 'global'})
            elif view_type == 'local':
                for idx, passage in enumerate(doc.get('passages', [])):
                    inputs = tokenizer(passage, return_tensors='pt', truncation=True, max_length=max_input_length).to(device)
                    output = model.generate(**inputs, max_length=max_output_length, num_return_sequences=1)
                    query = tokenizer.decode(output[0], skip_special_tokens=True)
                    results.append({'doc_id': doc_id, 'passage_id': idx, 'query': query, 'view': 'local'})
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--view_type', type=str, choices=['global', 'local'], required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)

    documents = load_documents(args.doc_path)
    queries = generate_queries(model, tokenizer, documents, args.view_type, device=args.device)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in queries:
            f.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    main()
