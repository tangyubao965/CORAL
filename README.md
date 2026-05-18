# CORAL: Corpus-Aligned Generative Retrieval with View-Aligned Modeling

This repository provides the implementation of **CORAL**, a corpus-aligned framework for **Generative Retrieval (GR)**. CORAL improves corpus memorization by aligning three key components of the GR pipeline: multi-view docid construction, corpus-aware training, and view-aware decoding.

Generative retrieval formulates retrieval as a sequence-to-sequence task, where the model directly generates document identifiers, i.e., docids, from natural language queries. Existing GR methods often focus on improving query-to-docid relevance, while CORAL emphasizes how corpus information is encoded and differentiated during indexing. The core idea is that effective GR requires docids that reflect the semantic structure of the corpus, objectives that distinguish document-docid relationships, and decoding mechanisms that adapt to different semantic views.

## Overview

CORAL consists of three main components.

### Hierarchical multi-view docids

CORAL constructs docids from multiple semantic views of a document. Global docids capture document-level semantics, while local docids capture fine-grained passage-level semantics. Title-based docids are also used as document-level signals. This design allows the model to memorize both broad document topics and more specific local evidence.

### Group-wise contrastive learning

CORAL treats each document and its associated docids as a semantic group. The group-wise contrastive objective encourages docids from the same document group to be closer in representation space, while separating docids from different documents. This helps the model build a more discriminative corpus memory.

### View-aware decoding

CORAL introduces view-specific LoRA adapters and a gating network into the decoder. The adapters provide lightweight view-specific residual updates, while the gating network dynamically controls how much the decoder relies on global or local decoding behavior. This allows the generation process to better adapt to different semantic granularities.

## Method Figure

Please place the method figure under `figures/` and update the path below if needed.

<p align="center">
  <img src="figures/coral_overview.png" width="850">
</p>

## Key Findings

Our experiments show that:

- CORAL consistently improves over strong generative retrieval baselines on MS MARCO 320K and NQ 320K.
- CORAL remains competitive against recent traditional retrieval baselines, including sparse and dense retrieval methods.
- The improvements come from the alignment of docid construction, training objective, and decoding structure, rather than simply increasing the number of docids.
- Hierarchical multi-view docids help the model capture both document-level and passage-level semantics.
- Group-wise contrastive learning improves the discriminative ability of corpus representations.
- View-aware decoding improves retrieval by adapting generation to the semantic scope of different docids.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── config.py
├── dataset.py
├── model.py
├── optimizer.py
├── train.py
├── inference.py
├── metrics.py
├── utils.py
├── analyze_visualization.py
│
├── preprocess/
│   ├── build_dataset.py
│   ├── build_doc_docid_pairs.py
│   ├── build_docid_dict.py
│   ├── build_docid_view.py
│   ├── build_labeled_query_docid_pairs.py
│   ├── build_prefix_trie.py
│   ├── build_qg_training_data.py
│   ├── generate_pseudo_queries.py
│   ├── passage_sampling_and_local_view.py
│   ├── preprocess_beir.py
│   ├── process_marco.py
│   └── train_query_generation_model.py
│
└── data/
    └── .gitkeep
```

The `data/` directory is intentionally left empty because the datasets are large. Please download the datasets manually or with the scripts below.

## Installation

We recommend using a clean Python environment.

```bash
conda create -n coral python=3.10
conda activate coral
pip install -r requirements.txt
```

Some dataset download examples below use additional packages:

```bash
pip install ir_datasets beir
```

## Data Preparation

This repository supports MS MARCO, NQ 320K, and BEIR-style datasets. Since the released `data/` folder is empty, please first download the required datasets and place them under `data/`.

After downloading, the expected directory structure is:

```text
data/
├── msmarco_data/
│   ├── collection.tsv
│   ├── queries.train.tsv
│   ├── queries.dev.tsv
│   └── qrels.train.tsv
│
├── nq/
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── qrels/
│       ├── train.tsv
│       ├── dev.tsv
│       └── test.tsv
│
└── beir/
    └── dataset_name/
        ├── corpus.jsonl
        ├── queries.jsonl
        └── qrels/
            ├── train.tsv
            ├── dev.tsv
            └── test.tsv
```

The processed files used by CORAL are stored under `Data/processed/`.

## Downloading MS MARCO

MS MARCO is not included in this repository. Please download the passage ranking files from the official MS MARCO dataset page.

Create the directory:

```bash
mkdir -p data/msmarco_data
```

A typical MS MARCO passage ranking setup requires:

```text
collection.tsv
queries.train.tsv
queries.dev.tsv
qrels.train.tsv
qrels.dev.tsv
```

You can download these files from the official MS MARCO ranking dataset page and place them under `Data/msmarco_data/`.

If your downloaded files are compressed, unzip or untar them first. For example:

```bash
tar -xzf collection.tar.gz -C data/msmarco_data
tar -xzf queries.tar.gz -C data/msmarco_data
tar -xzf qrels.tar.gz -C data/msmarco_data
```

Then run the MS MARCO preprocessing script:

```bash
python preprocess/process_marco.py \
  --input_dir data/msmarco_data \
  --output_dir data/processed/msmarco320k
```

The preprocessing script converts raw MS MARCO files into the JSONL and qrels format used by CORAL.

## Downloading NQ 320K

The NQ 320K data used in retrieval experiments is derived from Natural Questions-style question-answering data. Because the raw data is large, it is not included in this repository.

Please obtain Natural Questions from an official source and convert it into the following retrieval format:

```text
data/nq/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
    ├── train.tsv
    ├── dev.tsv
    └── test.tsv
```

The required file formats are:

`corpus.jsonl`

```json
{"doc_id": "doc1", "title": "Document title", "text": "Document content"}
```

`queries.jsonl`

```json
{"query_id": "q1", "query": "Question text"}
```

`qrels/*.tsv`

```text
query_id    doc_id    relevance
q1          doc1      1
```

If you use `ir_datasets`, you can first cache the Natural Questions retrieval dataset locally:

```python
import ir_datasets

dataset = ir_datasets.load("natural-questions")
for query in dataset.queries_iter():
    pass
for doc in dataset.docs_iter():
    pass
for qrel in dataset.qrels_iter():
    pass
```

Then convert the loaded queries, documents, and qrels into the `Data/nq/` format above.

After preparing `data/nq/`, run:

```bash
python preprocess/build_dataset.py \
  --input_dir data/nq \
  --output_dir data/processed/nq
```

## Downloading BEIR

BEIR datasets are not included in this repository. To run CORAL on BEIR, download the desired BEIR dataset separately. The recommended way is to use the official BEIR package.

Create the directory:

```bash
mkdir -p data/beir
```

Install BEIR:

```bash
pip install beir
```

Download a BEIR dataset, for example `nfcorpus`:

```python
from beir import util

dataset = "nfcorpus"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"

out_dir = "data/beir"
data_path = util.download_and_unzip(url, out_dir)

print(data_path)
```

Common BEIR dataset names include:

```text
trec-covid
nfcorpus
nq
hotpotqa
fiqa
arguana
webis-touche2020
quora
dbpedia-entity
scidocs
fever
climate-fever
scifact
```

After downloading, the directory should look like:

```text
data/
└── beir/
    └── nfcorpus/
        ├── corpus.jsonl
        ├── queries.jsonl
        └── qrels/
            ├── train.tsv
            ├── dev.tsv
            └── test.tsv
```

Some BEIR datasets may only provide a subset of these splits. For example, if only the test split is available, use `--splits test`.

Run:

```bash
python preprocess/preprocess_beir.py \
  --beir_dir data/beir/nfcorpus \
  --output_dir data/processed/beir/nfcorpus \
  --splits train,dev,test
```

For a BEIR dataset with only a test split:

```bash
python preprocess/preprocess_beir.py \
  --beir_dir data/beir/fiqa \
  --output_dir data/processed/beir/fiqa \
  --splits test
```

## Preprocessing Pipeline

After obtaining a dataset and converting it into the expected format, run the CORAL preprocessing pipeline.

### 1. Build query generation training data

```bash
python preprocess/build_qg_training_data.py \
  --dataset_dir data/processed/msmarco320k \
  --output_dir data/qg_training/msmarco320k
```

### 2. Train the query generation model

```bash
python preprocess/train_query_generation_model.py \
  --train_file data/qg_training/msmarco320k/train.jsonl \
  --output_dir outputs/qg_model/msmarco320k
```

### 3. Generate pseudo-queries

```bash
python preprocess/generate_pseudo_queries.py \
  --model_dir outputs/qg_model/msmarco320k \
  --corpus_file data/processed/msmarco320k/corpus.jsonl \
  --output_file data/processed/msmarco320k/pseudo_queries.jsonl
```

### 4. Build global and local docid views

```bash
python preprocess/passage_sampling_and_local_view.py \
  --corpus_file data/processed/msmarco320k/corpus.jsonl \
  --output_file data/processed/msmarco320k/local_views.jsonl
```

```bash
python preprocess/build_docid_view.py \
  --corpus_file data/processed/msmarco320k/corpus.jsonl \
  --pseudo_query_file data/processed/msmarco320k/pseudo_queries.jsonl \
  --local_view_file data/processed/msmarco320k/local_views.jsonl \
  --output_dir data/processed/msmarco320k/docids
```

### 5. Build document-docid pairs

```bash
python preprocess/build_doc_docid_pairs.py \
  --docid_dir data/processed/msmarco320k/docids \
  --output_file data/processed/msmarco320k/doc_docid_pairs.jsonl
```

### 6. Build labeled query-docid pairs

```bash
python preprocess/build_labeled_query_docid_pairs.py \
  --qrels_file data/processed/msmarco320k/qrels/train.tsv \
  --docid_dir data/processed/msmarco320k/docids \
  --output_file data/processed/msmarco320k/train_pairs.jsonl
```

### 7. Build docid dictionary and prefix trie

```bash
python preprocess/build_docid_dict.py \
  --docid_dir data/processed/msmarco320k/docids \
  --output_dir data/processed/msmarco320k/index
```

```bash
python preprocess/build_prefix_trie.py \
  --docid_file data/processed/msmarco320k/index/valid_docids.txt \
  --output_file data/processed/msmarco320k/index/prefix_trie.pkl
```

## Training

Train CORAL with the processed query-docid pairs.

```bash
python train.py \
  --config config.py \
  --train_file data/processed/msmarco320k/train_pairs.jsonl \
  --docid_view_file data/processed/msmarco320k/docids/docid_view_labels.json \
  --output_dir outputs/checkpoints/coral_msmarco320k
```

The main model is implemented in `model.py`. It includes:

- sequence-to-sequence retrieval backbone
- hierarchical multi-view docid modeling
- group-wise contrastive learning
- view-specific LoRA adapters
- view gating network

The optimizer is implemented in `optimizer.py`.

## Inference

During inference, CORAL uses trie-constrained beam search to ensure that generated identifiers are valid corpus docids.

```bash
python inference.py \
  --checkpoint_path outputs/checkpoints/coral_msmarco320k/best.pt \
  --query_file data/processed/msmarco320k/queries_test.jsonl \
  --prefix_trie data/processed/msmarco320k/index/prefix_trie.pkl \
  --docid2doc data/processed/msmarco320k/index/docid2doc.json \
  --output_file outputs/predictions/coral_msmarco320k_test.json \
  --num_beams 100 \
  --top_k_docids 100 \
  --top_k_docs 100
```

The generated docids are mapped back to source documents. Since multiple docids can correspond to the same document, docid-level scores are aggregated into document-level scores.

## Evaluation

The evaluation metrics are implemented in `metrics.py`.

For MS MARCO 320K, we use:

- MRR@3
- Hits@1
- Hits@10
- Hits@100

For NQ 320K, we use:

- MRR@1
- Hits@1
- Hits@10
- MRR@20

Example:

```bash
python metrics.py \
  --qrels data/processed/msmarco320k/qrels/test.tsv \
  --run outputs/predictions/coral_msmarco320k_test.json \
  --metrics MRR@3,Hits@1,Hits@10,Hits@100
```

For NQ 320K:

```bash
python metrics.py \
  --qrels data/processed/nq/qrels/test.tsv \
  --run outputs/predictions/coral_nq_test.json \
  --metrics MRR@1,Hits@1,Hits@10,MRR@20
```

## Visualization

The repository includes `analyze_visualization.py` for visualizing query and docid representations.

```bash
python analyze_visualization.py \
  --input_file outputs/representations/coral_representations.jsonl \
  --output_file outputs/visualization/coral_tsne.png
```

This script can be used to inspect whether global and local docids form more structured semantic clusters.

## Output Format

The inference output is expected to contain ranked documents for each query.

```json
{
  "query_id": "q1",
  "ranked_docs": [
    {"doc": "doc1", "score": -0.12},
    {"doc": "doc2", "score": -0.35}
  ]
}
```

If generated docids are saved, the format is:

```json
{
  "query_id": "q1",
  "generated_docids": [
    {"docid": "global docid text", "score": -0.10},
    {"docid": "local docid text", "score": -0.18}
  ]
}
```

The docid-to-document mapping is stored as:

```json
{
  "global docid text": "doc1",
  "local docid text": "doc1"
}
```
