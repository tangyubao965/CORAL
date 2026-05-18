"""Inference for CORAL with trie-constrained beam search and count-normalized aggregation.

This script generates multiple valid docids for each query, maps them back to
source documents, and aggregates docid-level scores into document-level scores:

    score(q, d) = mean_{id in I_q(d)} s(q, id),
    s(q, id) = log p(id | q) / |id|.

The prefix trie is built from all valid corpus docids. The root node corresponds
to the decoder BOS token, and each child node represents a valid next token.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer

try:
    # Use the updated view-aware model.
    from model import CORALModel
except ImportError:
    # Fallback for older project names.
    from model import RetrievalModel as CORALModel  # type: ignore

from dataset import InferenceDataset


@dataclass
class TrieNode:
    """A node in the prefix trie.

    Each node is keyed by a token id. The root represents the decoder BOS token.
    A terminal node indicates that the path from the root to this node forms a
    complete valid docid.
    """

    token_id: Optional[int] = None
    children: Dict[int, "TrieNode"] = field(default_factory=dict)
    is_terminal: bool = False


class PrefixTrie:
    """Prefix trie over tokenized docids for constrained decoding."""

    def __init__(self, bos_token_id: int, eos_token_id: int) -> None:
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.root = TrieNode(token_id=bos_token_id)

    def insert(self, token_ids: Iterable[int]) -> None:
        """Insert one valid docid path.

        The input should not include BOS. EOS is appended if absent, so generation
        can terminate only after a complete docid.
        """
        ids = list(token_ids)
        if len(ids) == 0:
            return
        if ids[-1] != self.eos_token_id:
            ids.append(self.eos_token_id)

        node = self.root
        for token_id in ids:
            if token_id not in node.children:
                node.children[token_id] = TrieNode(token_id=token_id)
            node = node.children[token_id]
        node.is_terminal = True

    def _node_for_prefix(self, prefix_ids: List[int]) -> Optional[TrieNode]:
        """Return the trie node for the current generated prefix."""
        if len(prefix_ids) == 0:
            return self.root

        # HuggingFace generation starts with decoder_start_token_id. In our
        # description this corresponds to the BOS root, so we skip it when
        # traversing child nodes.
        if prefix_ids[0] == self.bos_token_id:
            prefix_ids = prefix_ids[1:]

        node = self.root
        for token_id in prefix_ids:
            if token_id not in node.children:
                return None
            node = node.children[token_id]
        return node

    def allowed_next_tokens(self, prefix_ids: List[int]) -> List[int]:
        """Return valid next-token candidates for a generated prefix."""
        node = self._node_for_prefix(prefix_ids)
        if node is None:
            return [self.eos_token_id]
        if len(node.children) == 0:
            return [self.eos_token_id]
        return list(node.children.keys())


def build_prefix_trie(
    valid_docids: Iterable[str],
    tokenizer: T5Tokenizer,
    bos_token_id: int,
    eos_token_id: int,
) -> PrefixTrie:
    """Build a prefix trie from all valid corpus docids."""
    trie = PrefixTrie(bos_token_id=bos_token_id, eos_token_id=eos_token_id)
    for docid in valid_docids:
        token_ids = tokenizer.encode(docid, add_special_tokens=False)
        trie.insert(token_ids)
    return trie


def normalize_doc_mapping(raw_mapping: Dict[str, Any]) -> Dict[str, str]:
    """Normalize docid-to-document mapping values to document ids.

    The input project sometimes stores doc text or a dict as the value. For
    aggregation, we only need a stable document key. If the value is a dict and
    contains `doc_id`, `docid`, or `id`, we use it; otherwise we serialize it.
    """
    normalized = {}
    for docid, value in raw_mapping.items():
        if isinstance(value, dict):
            doc_key = value.get("doc_id") or value.get("docid") or value.get("id")
            normalized[docid] = str(doc_key) if doc_key is not None else json.dumps(value, sort_keys=True)
        else:
            normalized[docid] = str(value)
    return normalized


def length_normalized_sequence_scores(
    generation_output: Any,
    model: torch.nn.Module,
    input_batch_size: int,
    num_return_sequences: int,
    pad_token_id: int,
    eos_token_id: int,
) -> torch.Tensor:
    """Compute length-normalized log probabilities for generated sequences.

    We use HuggingFace transition scores, sum the log probabilities of generated
    tokens, and divide by the number of non-special generated tokens. EOS is not
    counted in the length, matching the docid length used for ranking.
    """
    sequences = generation_output.sequences

    if hasattr(model, "model"):
        hf_model = model.model
    else:
        hf_model = model

    transition_scores = hf_model.compute_transition_scores(
        sequences=generation_output.sequences,
        scores=generation_output.scores,
        beam_indices=getattr(generation_output, "beam_indices", None),
        normalize_logits=True,
    )

    # transition_scores has shape [B * R, generated_length]. It excludes the
    # initial decoder start token. We compute valid lengths from the generated
    # part of the sequence.
    generated_part = sequences[:, 1: 1 + transition_scores.size(1)]
    valid_mask = (generated_part != pad_token_id) & (generated_part != eos_token_id)
    lengths = valid_mask.sum(dim=-1).clamp_min(1)
    scores = transition_scores.sum(dim=-1) / lengths

    return scores.view(input_batch_size, num_return_sequences)


def aggregate_doc_scores(
    generated_docids: List[str],
    docid_scores: List[float],
    docid2doc: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Aggregate docid-level scores into count-normalized document scores."""
    per_doc_scores: Dict[str, List[float]] = defaultdict(list)
    per_doc_docids: Dict[str, List[str]] = defaultdict(list)

    for docid, score in zip(generated_docids, docid_scores):
        if docid not in docid2doc:
            # This should rarely happen under trie constraints, but keeping this
            # guard makes the script robust to tokenization or mapping mismatch.
            continue
        doc_key = docid2doc[docid]
        per_doc_scores[doc_key].append(float(score))
        per_doc_docids[doc_key].append(docid)

    ranked_docs = []
    for doc_key, scores in per_doc_scores.items():
        ranked_docs.append(
            {
                "doc": doc_key,
                "score": sum(scores) / len(scores),
                "matched_docids": per_doc_docids[doc_key],
                "docid_scores": scores,
                "num_matched_docids": len(scores),
            }
        )

    ranked_docs.sort(key=lambda x: x["score"], reverse=True)
    return ranked_docs


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """Load the CORAL model checkpoint."""
    try:
        model = CORALModel(
            model_name=args.model_name,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            freeze_base_model=args.freeze_base_model,
            gate_loss_weight=args.gate_loss_weight,
        )
    except TypeError:
        # Fallback for older RetrievalModel constructor.
        model = CORALModel(model_name=args.model_name, adapter_hidden_size=args.adapter_hidden_size)

    state = torch.load(args.checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=args.strict_load)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="t5-base")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/model.pt")
    parser.add_argument("--query_path", type=str, default="data/query.json")
    parser.add_argument("--docid2doc_path", type=str, default="data/docid2doc.json")
    parser.add_argument("--output_path", type=str, default="outputs/inference_result.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=50)
    parser.add_argument("--top_k_docids", type=int, default=50)
    parser.add_argument("--top_k_docs", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--gate_loss_weight", type=float, default=1.0)
    parser.add_argument("--freeze_base_model", action="store_true")
    parser.add_argument("--adapter_hidden_size", type=int, default=64)
    parser.add_argument("--strict_load", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    raw_docid2doc = json.load(open(args.docid2doc_path, "r", encoding="utf-8"))
    docid2doc = normalize_doc_mapping(raw_docid2doc)

    model = load_model(args, device)

    if hasattr(model, "model"):
        config = model.model.config
    else:
        config = model.config

    bos_token_id = config.decoder_start_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.pad_token_id
    eos_token_id = config.eos_token_id or tokenizer.eos_token_id
    pad_token_id = config.pad_token_id or tokenizer.pad_token_id

    trie = build_prefix_trie(
        valid_docids=docid2doc.keys(),
        tokenizer=tokenizer,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
        return trie.allowed_next_tokens(input_ids.detach().cpu().tolist())

    inference_dataset = InferenceDataset(args.query_path, tokenizer)
    inference_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results: Dict[str, Any] = {}

    with torch.no_grad():
        for batch in inference_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            query_ids = batch["query_ids"]

            if hasattr(model, "generate_docids"):
                generation_output = model.generate_docids(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    num_beams=args.num_beams,
                    num_return_sequences=args.top_k_docids,
                    max_new_tokens=args.max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )
            else:
                generation_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    num_beams=args.num_beams,
                    num_return_sequences=args.top_k_docids,
                    max_new_tokens=args.max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )

            score_tensor = length_normalized_sequence_scores(
                generation_output=generation_output,
                model=model,
                input_batch_size=input_ids.size(0),
                num_return_sequences=args.top_k_docids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

            sequences = generation_output.sequences.view(input_ids.size(0), args.top_k_docids, -1)
            for i, query_id in enumerate(query_ids):
                docids = [
                    tokenizer.decode(seq, skip_special_tokens=True).strip()
                    for seq in sequences[i]
                ]
                docid_scores = score_tensor[i].detach().cpu().tolist()
                ranked_docs = aggregate_doc_scores(docids, docid_scores, docid2doc)

                results[str(query_id)] = {
                    "generated_docids": [
                        {"docid": docid, "score": float(score)}
                        for docid, score in zip(docids, docid_scores)
                    ],
                    "ranked_docs": ranked_docs[: args.top_k_docs],
                }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
