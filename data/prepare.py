"""Tokenize FineWeb-Edu (sample-10BT) with GPT-2 BPE into flat uint16 token files.

Output (next to this script by default):
  val.bin, train.bin  - raw np.uint16 tokens, documents separated by <|endoftext|>
  meta.json           - token counts and tokenizer info

The dataset is public; no Hugging Face token is needed. If you hit rate limits,
run `uv run hf auth login` once instead of putting a token in code.

Usage:
  uv run python data/prepare.py --train-tokens 100e6
"""

import argparse
import json
import sys
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def write_split(path, target, batches, enc, num_threads):
    eot = enc.eot_token
    written = 0
    with open(path, "wb") as f, tqdm(total=target, unit="tok", unit_scale=True, desc=os.path.basename(path)) as bar:
        while written < target:
            texts = next(batches)["text"]
            docs = enc.encode_ordinary_batch(texts, num_threads=num_threads)
            # Prefix every document with <|endoftext|> so the model learns document boundaries.
            arr = np.concatenate([np.array([eot] + d, dtype=np.uint16) for d in docs])
            arr = arr[: target - written]
            f.write(arr.tobytes())
            written += len(arr)
            bar.update(len(arr))
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--train-tokens", type=float, default=100e6)
    p.add_argument("--val-tokens", type=float, default=2e6)
    p.add_argument("--batch-docs", type=int, default=1024)
    args = p.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    assert enc.n_vocab < 2**16, "uint16 storage requires vocab < 65536"

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    batches = iter(ds.iter(batch_size=args.batch_docs))
    threads = os.cpu_count() or 4

    # Val first so it is a fixed, disjoint slice from the start of the stream.
    counts = {}
    for split, target in [("val", int(args.val_tokens)), ("train", int(args.train_tokens))]:
        counts[split] = write_split(os.path.join(args.out_dir, f"{split}.bin"), target, batches, enc, threads)

    meta = {
        "dataset": "HuggingFaceFW/fineweb-edu/sample-10BT",
        "tokenizer": "gpt2",
        "vocab_size": enc.n_vocab,
        "eot_token": enc.eot_token,
        "dtype": "uint16",
        "tokens": counts,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
    # The streaming dataset keeps a background thread prefetching more shards; on
    # some machines (seen on a 128-core pod) the interpreter then hangs at exit
    # waiting for it. All output is written and closed by now, so exit hard.
    sys.stdout.flush()
    os._exit(0)
