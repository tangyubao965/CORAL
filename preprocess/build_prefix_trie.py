import json
from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_end = False

class PrefixTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, docid):
        node = self.root
        for token in docid.split():
            node = node.children[token]
        node.is_end = True

    def save(self, path):
        def serialize(node):
            return {
                "end": node.is_end,
                "children": {k: serialize(v) for k, v in node.children.items()}
            }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialize(self.root), f, indent=2)

def build_trie_from_docids(docid_file, output_path):
    trie = PrefixTrie()
    with open(docid_file, "r", encoding="utf-8") as f:
        for line in f:
            docid = line.strip()
            if docid:
                trie.insert(docid)
    trie.save(output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--docid_file", type=str, required=True, help="Path to the file containing docids")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for serialized trie")
    args = parser.parse_args()

    build_trie_from_docids(args.docid_file, args.output_path)
