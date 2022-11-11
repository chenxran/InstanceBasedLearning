from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, BartTokenizer
from tqdm import tqdm

import multiprocessing as mp
import numpy as np
import shutil
from transformers.modeling_outputs import BaseModelOutput

lock = mp.RLock()


class RNNLogicGenerator(nn.Module):
    def __init__(self, args, num_relations, embedding_dim, hidden_dim, print=print):
        super(RNNLogicGenerator, self).__init__()
        self.args = args
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.mov = num_relations // 2
        self.vocab_size = self.num_relations + 2
        self.label_size = self.num_relations + 1
        self.ending_idx = num_relations
        self.padding_idx = self.num_relations + 1
        self.num_layers = 1
        self.use_cuda = True
        self.print = print

        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        self.rnn = torch.nn.LSTM(self.embedding_dim * 2, self.hidden_dim, self.num_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_dim, self.label_size)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        self.relation2id = {}
        self.id2relation = {}

        self.device = "cuda:1" if self.args["parallel"] else "cuda:0"

        with open(args['data_dir'] + '/relation2text.txt', 'r', encoding= 'utf-8') as reader:
            for i, line in enumerate(reader):
                if '\t' in line:
                    text = line.strip().split('\t')[1]
                else:
                    text = line.strip()
                self.relation2id[text] = i
                self.id2relation[i] = text

        self.pad_id = len(self.id2relation) + 1
        self.end_id = len(self.id2relation)

    def inv(self, r):
        if r < self.mov:
            return r + self.mov
        else:
            return r - self.mov

    def zero_state(self, batch_size):
        state_shape = (self.num_layers, batch_size, self.hidden_dim)
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return (h0.to(self.device), c0.to(self.device))
        else:
            return (h0, c0)

    def forward(self, inputs, relation, hidden):
        embedding = self.embedding(inputs)
        embedding_r = self.embedding(relation).unsqueeze(1).expand(-1, inputs.size(1), -1)
        embedding = torch.cat([embedding, embedding_r], dim=-1)
        outputs, hidden = self.rnn(embedding, hidden)
        logits = self.linear(outputs)
        # Predictor.clean()
        return logits, hidden

    def loss(self, inputs, target, mask, weight):
        if self.use_cuda:
            inputs = inputs.to(self.device)
            target = target.to(self.device)
            mask = mask.to(self.device)
            weight = weight.to(self.device)

        hidden = self.zero_state(inputs.size(0))
        logits, hidden = self.forward(inputs, inputs[:, 0], hidden)
        logits = torch.masked_select(logits, mask.unsqueeze(-1)).view(-1, self.label_size)
        target = torch.masked_select(target, mask)
        weight = torch.masked_select((mask.t() * weight).t(), mask)
        loss = (self.criterion(logits, target) * weight).sum() / weight.sum()
        return loss

    def sample(self, relation):
        rule = [relation]
        relation = torch.LongTensor([relation])
        if self.use_cuda:
            relation = relation.to(self.device)
        hidden = self.zero_state(1)
        while True:
            inputs = torch.LongTensor([[rule[-1]]])
            if self.use_cuda:
                inputs = inputs.to(self.device)
            logits, hidden = self.forward(inputs, relation, hidden)
            probability = torch.softmax(logits.squeeze(0).squeeze(0), dim=-1)
            sample = torch.multinomial(probability, 1).item()
            if sample == self.ending_idx:
                break
            rule.append(sample)
        return rule

    def log_probability(self, rule):
        rule.append(self.ending_idx)
        relation = torch.LongTensor([rule[0]])
        if self.use_cuda:
            relation = relation.to(self.device)
        hidden = self.zero_state(1)
        log_prob = 0.0
        for k in range(1, len(rule)):
            inputs = torch.LongTensor([[rule[k - 1]]])
            if self.use_cuda:
                inputs = inputs.to(self.device)
            logits, hidden = self.forward(inputs, relation, hidden)
            log_prob += torch.log_softmax(logits.squeeze(0).squeeze(0), dim=-1)[rule[k]]
        return log_prob

    def next_relation_log_probability(self, seq):
        inputs = torch.LongTensor([seq])
        relation = torch.LongTensor([seq[0]])
        if self.use_cuda:
            inputs = inputs.to(self.device)
            relation = relation.to(self.device)
        hidden = self.zero_state(1)
        logits, hidden = self.forward(inputs, relation, hidden)
        log_prob = torch.log_softmax(logits[0, -1, :] * 5, dim=-1).data.cpu().numpy().tolist()
        return log_prob

    def train_model(self, gen_batch, num_epoch=10000, lr=1e-3, print_epoch=100, relation=None):
        self.print("wait to train generator")
        if self.args["parallel"]:
            lock.acquire()
            print(f"lock acquired by relation {relation} for training generator.")

        self.to(self.device)
        self.train()
        # print = self.print
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epoch, eta_min=lr / 10)

        cum_loss = 0
        if gen_batch[0].size(0) == 0:
            num_epoch = 0
        for epoch in range(1, num_epoch + 1):

            loss = self.loss(*gen_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()

            cum_loss += loss.item()

            if epoch % print_epoch == 0:
                lr_str = "%.2e" % (opt.param_groups[0]['lr'])
                self.print(f"train_generator #{epoch} lr = {lr_str} loss = {cum_loss / print_epoch}")
                cum_loss = 0

        del opt
        del sch
        del gen_batch
        for p in self.parameters():
            del p.grad
        torch.cuda.empty_cache()

        self.cpu()
        if self.args["parallel"]:
            print(f"lock released by relation {relation}.")
            lock.release()

    def generate_rules(
        self, 
        relation: int, 
        num_samples: int, 
        max_len: int,
    ) -> Tuple[Tuple[Tuple[int], float]]:
        self.print("wait to generate")
        if self.args["parallel"]:
            lock.acquire()
            print(f"lock acquired by relation {relation} for generating rules.")

        self.to(self.device)
        self.eval()
        max_len += 1
        # print = self.print
        with torch.no_grad():
            found_rules = []
            prev_rules = [[[relation], 0]]
            for k in range(max_len):
                self.print(f"k = {k} |prev| = {len(prev_rules)}")
                current_rules = list()
                for _i, (rule, score) in enumerate(prev_rules):
                    assert rule[-1] != self.ending_idx
                    log_prob = self.next_relation_log_probability(rule)
                    for i in (range(self.label_size) if (k + 1) != max_len else [self.ending_idx]):
                        # if k != 0 and rule[-1] == self.inv(i):
                        # 	continue
                        new_rule = rule + [i]
                        new_score = score + log_prob[i]
                        (current_rules if i != self.ending_idx else found_rules).append((new_rule, new_score))

                # Predictor.clean()
                # if _i % 100 == 0:
                # 	pass
                # self.print(f"beam_search k = {k} i = {_i}")
                prev_rules = sorted(current_rules, key=lambda x: x[1], reverse=True)[:num_samples]
                found_rules = sorted(found_rules, key=lambda x: x[1], reverse=True)[:num_samples]

            self.print(f"generate |rules| = {len(found_rules)}")
            ret = ((rule[1:-1], score) for rule, score in found_rules)

            if not os.path.exists(f"{self.args['output_dir']}/generated_rules"):
                os.makedirs(f"{self.args['output_dir']}/generated_rules")
            with open(f"{self.args['output_dir']}/generated_rules/generated_rules_{relation}.txt", "w") as file:
                for rule, prob in ((rule[1:-1], score) for rule, score in found_rules):
                    file.write(f"{str(self.id2relation[relation])} -> {str([self.id2relation[_] for _ in rule if _ not in [self.pad_id, self.end_id]])} \t{str(round(prob, 4))}\n")

        self.cpu()
        if self.args["parallel"]:
            print(f"lock released by relation {relation}.")
            lock.release()
        return ret

