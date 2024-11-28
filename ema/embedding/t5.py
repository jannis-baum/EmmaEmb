import torch
import re
import numpy as np

from transformers import T5Tokenizer, T5EncoderModel
from torch.utils.data import DataLoader, Dataset

from ema.embedding.embedding_handler import EmbeddingHandler


class ProteinDataset(Dataset):
    def __init__(self, sequence_dict):
        self.protein_ids, self.sequences = zip(*sequence_dict.items())

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.protein_ids[idx], self.sequences[idx]


# Script kindly provided by Rost Lab
# https://github.com/agemagician/ProtTrans/blob/master/Embedding/prott5_embedder.py
class T5(EmbeddingHandler):
    def __init__(
        self,
        no_gpu: bool = False,
    ):
        super().__init__(no_gpu)
        self.model = None

    def get_embedding(
        self,
        model_id: str,
        protein_sequences: dict,
        output_dir: str,
        batch_size: int = 8,
    ):

        tokenizer = T5Tokenizer.from_pretrained(
            model_id, do_lower_case=False
        ).to(self.device)
        model = T5EncoderModel.from_pretrained(model_id).to(self.device)

        # only GPUs support half-precision currently; if you want to run on CPU use full-precision
        # (not recommended, much slower)
        (
            model.to(torch.float32)
            if self.device == torch.device("cpu")
            else model.half()
        )

        # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
        processed_sequences = {
            protein: " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
            for protein, sequence in protein_sequences.items()
        }

        # Create a dataset and dataloader
        dataset = ProteinDataset(protein_sequences)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch in dataloader:
            protein_ids, sequences = batch

            ids = tokenizer(
                list(sequences),
                return_tensors="pt",
                padding="longest",
                add_special_tokens=True,
            )
            input_ids = ids.input_ids.to(self.device)
            attention_mask = ids.attention_mask.to(self.device)

            with torch.no_grad():
                embedding_repr = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                # embeddings = outputs.last_hidden_state
