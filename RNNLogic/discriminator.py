from cmath import e
from typing import List

import numpy as np
import torch
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (BertConfig, BertForSequenceClassification, BertModel,
                          BertTokenizer)


class Discriminator(object):
    def __init__(self, path):
        super(Discriminator, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.config = BertConfig.from_pretrained(path)
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.model.eval()
        self.model.cuda()

    def calculate_prob(self, input_ids, attention_mask, token_type_ids):
        logits = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits

        # calculate probability from logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
        return probs.tolist()
    
    def train_model(self, premise, hypothesis, labels):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        inputs = self.tokenizer(
            premise, 
            hypothesis,
            max_length=128,
            padding="longest",
            truncation=True,
        )

        def collate_fn(features):
            batch = {}
            for k in features[0].keys():
                batch[k] = []
                for f in features:
                    batch[k].append(f[k])
                batch[k] = torch.tensor(batch[k])
            return batch


        inputs["labels"] = torch.tensor(labels)
        dataloader = DataLoader(
            Dataset.from_dict(inputs),
            batch_size=256,
            collate_fn=collate_fn,
        )

        with tqdm(total=len(dataloader) * 50) as pbar:
            for epoch in range(50):
                for batch in dataloader:
                    for k, v in batch.items():
                        batch[k] = v.cuda()
                    optimizer.zero_grad()
                    loss = self.model(**batch).loss
                    loss.backward()
                    optimizer.step()
                    pbar.set_description("Epoch: %d, Loss: %.4f" % (epoch, loss.item()))
                    pbar.update(1)
        self.model.eval()

    @torch.no_grad()
    def semantic_prob(self, premise: List[str], hypothesis: List[str], batch_size: int = 128) -> List[float]:
        probs = []
        for i in range(0, len(premise), batch_size):
            inputs = self.tokenizer(
                premise[i:i + batch_size], 
                hypothesis[i:i + batch_size],
                max_length=128,
                padding="longest",
                truncation=True,
                return_tensors='pt',
            )
            for k, v in inputs.items():
                inputs[k] = v.cuda()
            probs.extend(self.calculate_prob(**inputs))

        return probs