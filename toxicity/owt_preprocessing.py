import pickle

import datasets
from data import CONTEXT_LENGTH
from models import tokenizer
from transformer_lens.utils import tokenize_and_concatenate


def save_owt_data(tokenizer, ctx_length=CONTEXT_LENGTH, split="train"):
    dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train")
    if split == "train":
        # use 80% of the data
        dataset = dataset.select(range(int(0.8 * len(dataset))))
    elif split == "test":
        # use 20% of the data
        dataset = dataset.select(range(int(0.8 * len(dataset)), len(dataset)))
        print(len(dataset))
    else:
        raise ValueError("split must be either train or test")
    tokens_dataset = tokenize_and_concatenate(
        dataset,
        tokenizer,
        streaming=False,
        max_length=ctx_length,
        column_name="text",
        add_bos_token=True,
        num_proc=4,
    )
    with open(f"data/owt_{split}.pkl", "wb") as f:
        pickle.dump(tokens_dataset, f)


save_owt_data(tokenizer)
save_owt_data(tokenizer, split="test")
