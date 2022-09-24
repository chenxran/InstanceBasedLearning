#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset
import wandb

def construct_h_list(cls):
    cls.tail_batch_h_list = [{"h": [], "t": []} for _ in range(cls.nrelation)]
    for h, r, t in cls.train_triples:
        cls.tail_batch_h_list[r]["h"].append(h)
        cls.tail_batch_h_list[r]["t"].append(t)
    
    for key in range(cls.nrelation):
        cls.tail_batch_h_list[key]["h"] = torch.tensor(cls.tail_batch_h_list[key]["h"]).cuda()
        cls.tail_batch_h_list[key]["t"] = torch.tensor(cls.tail_batch_h_list[key]["t"]).cuda()

    cls.head_batch_h_list = [{"h": [], "t": []} for _ in range(cls.nrelation)]
    for h, r, t in cls.train_triples:
        cls.head_batch_h_list[r]["h"].append(t)
        cls.head_batch_h_list[r]["t"].append(h)
    
    for key in range(cls.nrelation):
        cls.head_batch_h_list[key]["h"] = torch.tensor(cls.head_batch_h_list[key]["h"]).cuda()
        cls.head_batch_h_list[key]["t"] = torch.tensor(cls.head_batch_h_list[key]["t"]).cuda()


ACTFN = {
    "none": lambda x: x,
    "relu": torch.relu,
    "tanh": torch.tanh,
}

class KGEModel(nn.Module):
    def __init__(self, args, model_name, nentity, nrelation, hidden_dim, gamma, weight, activation,
                 double_entity_embedding=False, double_relation_embedding=False, train_triples=None):
        super(KGEModel, self).__init__()
        self.args = args
        self.train_triples = train_triples
        self.activation = activation
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        self.weight = weight
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )

        self.flag = self.model_name not in ["CIBLERotatE", 'IBLErRotatE', 'CIBLErRotatE', 'IBLErRotatE']
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim), requires_grad=not args.freeze_entity_embedding)
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=not args.freeze_relation_embedding)
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'CIBLERotatE', 'IBLERotatE', 'CIBLErRotatE', 'IBLErRotatE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name in ['RotatE', 'CIBLERotatE', 'CIBLE', 'CIBLErRotatE', 'IBLErRotatE'] and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
        if model_name in ['CIBLERotatE', 'IBLERotatE', 'CIBLErRotatE', 'IBLErRotatE']:
            construct_h_list(self)

            if model_name in ['CIBLErRotatE', 'IBLErRotatE']:
                self.head_m_r = nn.Parameter(torch.randn(nrelation, self.entity_dim))
                nn.init.uniform_(
                    tensor=self.head_m_r, 
                    a=-self.embedding_range.item(), 
                    b=self.embedding_range.item()
                )
                self.tail_m_r = nn.Parameter(torch.randn(nrelation, self.entity_dim))
                nn.init.uniform_(
                    tensor=self.tail_m_r, 
                    a=-self.embedding_range.item(), 
                    b=self.embedding_range.item()
                )

        if self.args.mlp:
            if self.args.intermediate_dim is None:
                self.intermediate_dim = 2 * self.nentity
            else:
                self.intermediate_dim = self.args.intermediate_dim
            self.mlp = nn.Sequential(
                nn.Linear(self.nentity, self.intermediate_dim),
                nn.Tanh(),
                nn.Linear(self.intermediate_dim, self.nentity),
            )

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            relation_id = sample[:,1]
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            candidates = sample[:, 2]

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            candidates = head_part.view(-1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            relation_id = tail_part[:, 1]
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            relation_id = head_part[:, 1]
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            candidates = tail_part.view(-1)
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'CIBLERotatE': self.CIBLERotatE,
            'IBLERotatE': self.IBLERotatE,
            'CIBLErRotatE': self.CIBLErRotatE,
        }
        
        if self.model_name in model_func:
            if self.model_name in ['CIBLERotatE', 'CIBLE']:
                score = model_func[self.model_name](head, relation, tail, relation_id, candidates, mode)
            else:
                score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(p=2, dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def CIBLERotatE(self, head, relation, tail, relation_id, candidates, mode):
        batch_size = int(relation.size(0))
        negative_sample_size = int(candidates.size(0) / batch_size)

        score = self.RotatE(head, relation, tail, mode)
        
        identity_matrix_score = []
        identity_matrix = self.IBLERotatE(head, relation, tail, relation_id, mode)
        candidates = candidates.view(batch_size, negative_sample_size)
        for i in range(batch_size):
            identity_matrix_score.append(identity_matrix[i, candidates[i]])
        identity_matrix_score = torch.vstack(identity_matrix_score)

        return score, identity_matrix_score

    def IBLERotatE(self, head, relation, tail, relation_id, mode):
        pi = 3.14159265358979323846

        mask = torch.cuda.LongTensor(relation_id.size(0), self.nentity) * 0
        batch_indices, candidate_t_indices, h_indices = list(), list(), list()
        for i, _r in enumerate(relation_id):
            if mode == "head-batch":
                dictionary = self.head_batch_h_list[_r]
            else:
                dictionary = self.tail_batch_h_list[_r]

            candidate_t_indices.append(dictionary["t"])
            h_indices.append(dictionary["h"])
            if len(dictionary["h"]) != 0:
                mask[i][dictionary["h"].unique()] = 1

            length =len(dictionary["t"])
            batch_indices.append(torch.full(size=(length, ), fill_value=i, dtype=torch.long, device=relation.device))

        candidate_t_indices, h_indices, batch_indices = torch.cat(candidate_t_indices), torch.cat(h_indices), torch.cat(batch_indices)

        if mode == "head-batch":
            e1 = tail
        else:
            e1 = head

        e2 = self.entity_embedding[h_indices.unique().long(), :].unsqueeze(0)
        
        re_e1, im_e1 = torch.chunk(e1, 2, dim=2)
        re_e2, im_e2 = torch.chunk(e2, 2, dim=2)

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_e1 = re_relation * re_e1 + im_relation * im_e1
            im_e1 = re_relation * im_e1 - im_relation * re_e1

            re_e2 = re_relation * re_e2 + im_relation * im_e2
            im_e2 = re_relation * im_e2 - im_relation * re_e2

        else:
            re_e1 = re_e1 * re_relation - im_e1 * im_relation
            im_e1 = re_e1 * im_relation + im_e1 * re_relation

            re_e2 = re_e2 * re_relation - im_e2 * im_relation
            im_e2 = re_e2 * im_relation + im_e2 * re_relation

        re_score = re_e1 - re_e2
        im_score = im_e1 - im_e2

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(p=2, dim = 0)
        score = torch.sigmoid(self.gamma.item() - score.sum(dim = 2))

        identity_matrix = torch.cuda.FloatTensor(e1.size(0), self.nentity) * 0.
        identity_matrix[:, h_indices.unique().long()] = score  # (batch_size, nentity)
        identity_matrix[mask == 0] = -1e9

        selected_score = identity_matrix[batch_indices.long(), h_indices.long()]  # len(batch_indices)

        # selected_score[i] -> scores[batch_indices[i], candidate_t_indices[i], h_indices[i]]
        sparse_selected_scores = torch.sparse_coo_tensor(
            values=selected_score,
            indices=torch.vstack([batch_indices, candidate_t_indices, h_indices]),
            size=(len(e1), self.nentity, self.nentity),
        )

        if self.args.pooling == "mean":
            sparse_selected_counts = torch.sparse_coo_tensor(
                values=(selected_score / selected_score).detach().long(),
                indices=torch.vstack([batch_indices, candidate_t_indices, h_indices]),
                size=(len(e1), self.nentity, self.nentity),
            )
            scores = torch.sparse.sum(sparse_selected_scores, dim=2).to_dense()
            mask = scores == 0
            scores = scores / torch.sparse.sum(sparse_selected_counts, dim=2).to_dense()

            scores[mask] = 0.
        elif self.args.pooling == "sum":
            scores = torch.sparse.sum(sparse_selected_scores, dim=2).to_dense()

        return scores

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score

    def CIBLErRotatE(self, head, relation, tail, relation_id, candidates, mode):
        # batch_size = int(relation.size(0))
        # negative_sample_size = int(candidates.size(0) / batch_size)

        if mode == 'head-batch':
            head = head * self.head_m_r
            tail = tail * self.head_m_r
        else:
            head = head * self.tail_m_r
            tail = tail * self.tail_m_r

        return self.CIBLERotatE(head, relation, tail, relation_id, candidates, mode)
        # score = self.RotatE(head, relation, tail, mode)
        
        # identity_matrix_score = []
        # identity_matrix = self.IBLERotatE(head, relation, tail, relation_id, mode)
        # candidates = candidates.view(batch_size, negative_sample_size)
        # for i in range(batch_size):
        #     identity_matrix_score.append(identity_matrix[i, candidates[i]])
        # identity_matrix_score = torch.vstack(identity_matrix_score)

        # return score, identity_matrix_score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()
        
        for _ in range(args.gradient_accumulation_steps):
            positive_sample, negative_sample, subsampling_weight, mode, label = next(train_iterator)

            if model.flag:
                if args.cuda:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    subsampling_weight = subsampling_weight.cuda()

                negative_score = model((positive_sample, negative_sample), mode=mode)

                if args.negative_adversarial_sampling:
                    #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                    negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                                    * F.logsigmoid(-negative_score)).sum(dim = 1)
                else:
                    negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

                positive_score = model(positive_sample)
                positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

                if args.uni_weight:
                    positive_sample_loss = - positive_score.mean()
                    negative_sample_loss = - negative_score.mean()
                else:
                    positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
                    negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

                loss = (positive_sample_loss + negative_sample_loss) / 2
            else:
                if args.cuda:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    subsampling_weight = subsampling_weight.cuda()

                negative_score, negative_identity_matrix_score = model((positive_sample, negative_sample), mode=mode)
                negative_score = (model.weight * torch.sigmoid(negative_score) + negative_identity_matrix_score) / (1 + model.weight)

                positive_score, positive_identity_matrix_score = model(positive_sample)
                positive_score = (model.weight * torch.sigmoid(positive_score) + positive_identity_matrix_score) / (1 + model.weight)

                scores = torch.cat([positive_score, negative_score], dim=1) / model.args.temperature
                labels = (torch.cuda.FloatTensor(scores.size(0)) * 0.).long()

                loss = F.cross_entropy(scores, target=labels)

            if args.regularization != 0.0:
                #Use L3 regularization for ComplEx and DistMult
                regularization = args.regularization * (
                    model.entity_embedding.norm(p = 3)**3 + 
                    model.relation_embedding.norm(p = 3).norm(p = 3)**3
                )
                loss = loss + regularization
                regularization_log = {'regularization': regularization.item()}
            else:
                regularization_log = {}
            
            (loss / args.gradient_accumulation_steps).backward()

        optimizer.step()

        if model.flag:
            log = {
                **regularization_log,
                'positive_sample_loss': positive_sample_loss.item(),
                'negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item()
            }
        else:
            log = {
                **regularization_log,
                'loss': loss.item()
            }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        if model.flag:
                            score = model((positive_sample, negative_sample), mode)        
                        else:
                            score, identity_matrix_score = model((positive_sample, negative_sample), mode)
                            score = (model.weight * torch.sigmoid(score) + identity_matrix_score) / (1 + model.weight)

                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics


# class KGEV2Model(nn.Module):
#     def __init__(self, args, model_name, nentity, nrelation, hidden_dim, gamma, weight, activation,
#                  double_entity_embedding=False, double_relation_embedding=False, train_triples=None):
#         super(KGEV2Model, self).__init__()
#         self.args = args
#         self.train_triples = train_triples
#         self.activation = activation
#         self.model_name = model_name
#         self.nentity = nentity
#         self.nrelation = nrelation
#         self.hidden_dim = hidden_dim
#         self.epsilon = 2.0
        
#         self.gamma = nn.Parameter(
#             torch.Tensor([gamma]), 
#             requires_grad=False
#         )
#         self.weight = weight
        
#         self.embedding_range = nn.Parameter(
#             torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
#             requires_grad=False
#         )
        
#         self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
#         self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
#         self.entity_embedding = nn.Parameter(torch.randn(nentity, self.entity_dim), requires_grad=not args.freeze_entity_embedding)
#         nn.init.uniform_(
#             tensor=self.entity_embedding, 
#             a=-self.embedding_range.item(), 
#             b=self.embedding_range.item()
#         )
#         # torch.nn.init.xavier_uniform_(self.entity_embedding)
#         self.relation_embedding = nn.Parameter(torch.randn(nrelation, self.relation_dim), requires_grad=not args.freeze_relation_embedding)
#         nn.init.uniform_(
#             tensor=self.relation_embedding, 
#             a=-self.embedding_range.item(), 
#             b=self.embedding_range.item()
#         )
#         # torch.nn.init.xavier_uniform_(self.relation_embedding)
        
#         if model_name == 'pRotatE':
#             self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
#         #Do not forget to modify this line when you add a new model in the "forward" function
#         if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'CIBLERotatE', 'CIBLE', 'TransR', 'CIBLETransR']:
#             raise ValueError('model %s not supported' % model_name)
            
#         if model_name in ['RotatE', 'CIBLERotatE', 'CIBLE'] and (not double_entity_embedding or double_relation_embedding):
#             raise ValueError('RotatE should use --double_entity_embedding')

#         if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
#             raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
#         if model_name in ['CIBLERotatE', 'CIBLE']:
#             construct_h_list(self)
#             self.head_m_r = nn.Parameter(torch.randn(nrelation, self.entity_dim, self.entity_dim))
#             nn.init.uniform_(
#                 tensor=self.head_m_r, 
#                 a=-self.embedding_range.item(), 
#                 b=self.embedding_range.item()
#             )

#             self.tail_m_r = nn.Parameter(torch.randn(nrelation, self.entity_dim, self.entity_dim))
#             nn.init.uniform_(
#                 tensor=self.tail_m_r, 
#                 a=-self.embedding_range.item(), 
#                 b=self.embedding_range.item()
#             )

#         if self.args.mlp:
#             if self.args.intermediate_dim is None:
#                 self.intermediate_dim = 2 * self.nentity
#             else:
#                 self.intermediate_dim = self.args.intermediate_dim
#             self.mlp = nn.Sequential(
#                 nn.Linear(self.nentity, self.intermediate_dim),
#                 nn.Tanh(),
#                 nn.Dropout(p=0.1),
#                 nn.Linear(self.intermediate_dim, self.nentity),
#             )

#         if model_name in ['CIBLETransR', 'TransR']:
#             construct_h_list(self)
#             self.m_r = nn.Parameter(torch.randn(nrelation, self.entity_dim, self.entity_dim))
#             torch.nn.init.xavier_uniform_(self.m_r)

#     def forward(self, sample, mode='single'):
#         '''
#         Forward function that calculate the score of a batch of triples.
#         In the 'single' mode, sample is a batch of triple.
#         In the 'head-batch' or 'tail-batch' mode, sample consists two part.
#         The first part is usually the positive sample.
#         And the second part is the entities in the negative samples.
#         Because negative samples and positive samples usually share two elements 
#         in their triple ((head, relation) or (relation, tail)).
#         '''

#         if mode == 'head-batch':
            
#             head = self.entity_embedding.unsqueeze(0)

#             relation = torch.index_select(
#                 self.relation_embedding, 
#                 dim=0, 
#                 index=sample[:, 1]
#             ).unsqueeze(1)
#             relation_id = sample[:, 1]
            
#             tail = torch.index_select(
#                 self.entity_embedding, 
#                 dim=0, 
#                 index=sample[:, 2]
#             ).unsqueeze(1)
            
#         elif mode == 'tail-batch':
            
#             head = torch.index_select(
#                 self.entity_embedding, 
#                 dim=0,
#                 index=sample[:, 0]
#             ).unsqueeze(1)
            
#             relation = torch.index_select(
#                 self.relation_embedding,
#                 dim=0,
#                 index=sample[:, 1]
#             ).unsqueeze(1)
#             relation_id = sample[:, 1]

#             tail = self.entity_embedding.unsqueeze(0)
            
#         else:
#             raise ValueError('mode %s not supported' % mode)
            
#         model_func = {
#             'TransE': self.TransE,
#             'DistMult': self.DistMult,
#             'ComplEx': self.ComplEx,
#             'RotatE': self.RotatE,
#             'pRotatE': self.pRotatE,
#             'CIBLERotatE': self.CIBLERotatE,
#             'CIBLE': self.CIBLE,
#             # 'TransR': self.TransR,
#             # 'CIBLETransR': self.CIBLETransR,
#         }
        
#         if self.model_name in model_func:
#             if self.model_name in ['CIBLERotatE', 'CIBLE']:
#                 score = model_func[self.model_name](head, relation, tail, relation_id, mode)
#             elif self.model_name in ['TransR', 'CIBLETransR']:
#                 score = model_func[self.model_name](head, relation, tail, relation_id, mode)
#             else:
#                 score = model_func[self.model_name](head, relation, tail, mode)
#         else:
#             raise ValueError('model %s not supported' % self.model_name)
        
#         return score
    
#     def TransE(self, head, relation, tail, mode):
#         if mode == 'head-batch':
#             score = head + (relation - tail)
#         else:
#             score = (head + relation) - tail

#         score = self.gamma.item() - torch.norm(score, p=1, dim=2)
#         return score

#     def DistMult(self, head, relation, tail, mode):
#         if mode == 'head-batch':
#             score = head * (relation * tail)
#         else:
#             score = (head * relation) * tail

#         score = score.sum(dim = 2)
#         return score

#     def ComplEx(self, head, relation, tail, mode):
#         re_head, im_head = torch.chunk(head, 2, dim=2)
#         re_relation, im_relation = torch.chunk(relation, 2, dim=2)
#         re_tail, im_tail = torch.chunk(tail, 2, dim=2)

#         if mode == 'head-batch':
#             re_score = re_relation * re_tail + im_relation * im_tail
#             im_score = re_relation * im_tail - im_relation * re_tail
#             score = re_head * re_score + im_head * im_score
#         else:
#             re_score = re_head * re_relation - im_head * im_relation
#             im_score = re_head * im_relation + im_head * re_relation
#             score = re_score * re_tail + im_score * im_tail

#         score = score.sum(dim = 2)
#         return score

#     def RotatE(self, head, relation, tail, mode):
#         pi = 3.14159265358979323846
        
#         re_head, im_head = torch.chunk(head, 2, dim=2)
#         re_tail, im_tail = torch.chunk(tail, 2, dim=2)

#         #Make phases of relations uniformly distributed in [-pi, pi]

#         phase_relation = relation/(self.embedding_range.item()/pi)

#         re_relation = torch.cos(phase_relation)
#         im_relation = torch.sin(phase_relation)

#         if mode == 'head-batch':
#             re_score = re_relation * re_tail + im_relation * im_tail
#             im_score = re_relation * im_tail - im_relation * re_tail
#             re_score = re_score - re_head
#             im_score = im_score - im_head
#         else:
#             re_score = re_head * re_relation - im_head * im_relation
#             im_score = re_head * im_relation + im_head * re_relation
#             re_score = re_score - re_tail
#             im_score = im_score - im_tail

#         score = torch.stack([re_score, im_score], dim = 0)
#         score = score.norm(p=2, dim = 0)

#         score = self.gamma.item() - score.sum(dim = 2)
#         return score, score, score

#     def CIBLERotatE(self, head, relation, tail, relation_id, mode):
#         pi = 3.14159265358979323846

#         batch_size = int(relation.size(0))
#         # negative_sample_size = int(candidates.size(0) / batch_size)

#         re_head, im_head = torch.chunk(head, 2, dim=2)
#         re_tail, im_tail = torch.chunk(tail, 2, dim=2)

#         #Make phases of relations uniformly distributed in [-pi, pi]

#         phase_relation = relation/(self.embedding_range.item()/pi)

#         re_relation = torch.cos(phase_relation)
#         im_relation = torch.sin(phase_relation)

#         if mode == 'head-batch':
#             re_score = re_relation * re_tail + im_relation * im_tail
#             im_score = re_relation * im_tail - im_relation * re_tail

#             re_e1 = re_score.clone()
#             im_e1 = im_score.clone()

#             re_score = re_score - re_head
#             im_score = im_score - im_head
#         else:
#             re_score = re_head * re_relation - im_head * im_relation
#             im_score = re_head * im_relation + im_head * re_relation

#             re_e1 = re_score.clone()
#             im_e1 = im_score.clone()

#             re_score = re_score - re_tail
#             im_score = im_score - im_tail

#         score = torch.stack([re_score, im_score], dim = 0)
#         score = score.norm(p=2, dim = 0)
#         score = self.gamma.item() - score.sum(dim = 2)

#         identity_matrix_score = self.identity_matrix(head, relation, tail, relation_id, mode, re_e1=re_e1, im_e1=im_e1)

#         return score * self.args.weight + identity_matrix_score, score, identity_matrix_score

#     def CIBLE(self, head, relation, tail, relation_id, mode):
#         identity_matrix_score = self.identity_matrix(head, relation, tail, relation_id, mode)
#         return identity_matrix_score, None, None

#     def identity_matrix(self, head, relation, tail, relation_id, mode, re_e1=None, im_e1=None):
#         pi = 3.14159265358979323846

#         mask = torch.cuda.LongTensor(relation_id.size(0), self.nentity) * 0
#         batch_indices, candidate_t_indices, h_indices = list(), list(), list()
#         for i, _r in enumerate(relation_id):
#             if mode == "head-batch":
#                 dictionary = self.head_batch_h_list[_r]
#             else:
#                 dictionary = self.tail_batch_h_list[_r]

#             candidate_t_indices.append(dictionary["t"])
#             h_indices.append(dictionary["h"])
#             if len(dictionary["h"]) != 0:
#                 mask[i][dictionary["h"].unique()] = 1

#             length =len(dictionary["t"])
#             batch_indices.append(torch.full(size=(length, ), fill_value=i, dtype=torch.long, device=relation.device))

#         candidate_t_indices, h_indices, batch_indices = torch.cat(candidate_t_indices), torch.cat(h_indices), torch.cat(batch_indices)

#         if mode == "head-batch":
#             e1 = tail
#             rel = self.head_m_r[relation_id].view(-1, self.entity_dim, self.entity_dim)
#         else:
#             e1 = head
#             rel = self.tail_m_r[relation_id].view(-1, self.entity_dim, self.entity_dim)
#         e2 = self.entity_embedding[h_indices.unique().long(), :].unsqueeze(0)

#         e1_score = e1 @ rel
#         e2_score = e2 @ rel

#         if self.args.normalize:
#             e1_score = (e1_score / e1_score.norm(dim=2, p=2, keepdim=True))
#             e2_score = (e2_score / e2_score.norm(dim=2, p=2, keepdim=True))

#         identity_matrix = torch.cuda.FloatTensor(e1_score.size(0), self.nentity) * 0.
#         identity_matrix[:, h_indices.unique().long()] = ACTFN[self.activation](torch.bmm(e1_score, e2_score.permute(0, 2, 1).contiguous()).squeeze(1))

#         selected_score = identity_matrix[batch_indices.long(), h_indices.long()]  # len(batch_indices)
#         scores = torch.sparse.sum(
#             torch.sparse_coo_tensor(
#                 values=selected_score,
#                 indices=torch.vstack([batch_indices, candidate_t_indices, h_indices]),
#                 size=(len(e1), self.nentity, self.nentity),
#             ),
#             dim=2,
#         ).to_dense()

#         if self.args.mlp:
#             scores = self.mlp(scores)
#         return scores

#     def pRotatE(self, head, relation, tail, mode):
#         pi = 3.14159262358979323846
        
#         #Make phases of entities and relations uniformly distributed in [-pi, pi]

#         phase_head = head/(self.embedding_range.item()/pi)
#         phase_relation = relation/(self.embedding_range.item()/pi)
#         phase_tail = tail/(self.embedding_range.item()/pi)

#         if mode == 'head-batch':
#             score = phase_head + (phase_relation - phase_tail)
#         else:
#             score = (phase_head + phase_relation) - phase_tail

#         score = torch.sin(score)            
#         score = torch.abs(score)

#         score = self.gamma.item() - score.sum(dim = 2) * self.modulus
#         return score
    
#     @staticmethod
#     def train_step(model, optimizer, train_iterator, args):
#         '''
#         A single train step. Apply back-propation and return the loss
#         '''

#         model.train()

#         optimizer.zero_grad()
        
#         for _ in range(args.gradient_accumulation_steps):
#             positive_sample, negative_sample, subsampling_weight, mode, labels = next(train_iterator)

#             if args.cuda:
#                 positive_sample = positive_sample.cuda()
#                 labels = labels.cuda()

#             scores, rotate_score, identity_matrix_score = model(positive_sample, mode=mode)  # batch_size, nentity

#             loss = F.cross_entropy(scores, target=labels)
#             rotate_loss = F.cross_entropy(rotate_score, target=labels)
#             identity_matrix_loss = F.cross_entropy(identity_matrix_score, target=labels)
            
#             if args.regularization != 0.0:
#                 #Use L3 regularization for ComplEx and DistMult
#                 regularization = args.regularization * (
#                     model.entity_embedding.norm(p = 3)**3 + 
#                     model.relation_embedding.norm(p = 3).norm(p = 3)**3
#                 )
#                 loss = loss + regularization
#                 regularization_log = {'regularization': regularization.item()}
#             else:
#                 regularization_log = {}
            
#             (loss / args.gradient_accumulation_steps).backward()
#         optimizer.step()

#         log = {
#             **regularization_log,
#             'loss': loss.item(),
#             'rotate_loss': rotate_loss.item(),
#             'identity_matrix_loss': identity_matrix_loss.item(), 
#         }

#         return log
    
#     @staticmethod
#     def test_step(model, test_triples, all_true_triples, args):
#         '''
#         Evaluate the model on test or valid datasets
#         '''
        
#         model.eval()
        
#         if args.countries:
#             #Countries S* datasets are evaluated on AUC-PR
#             #Process test data for AUC-PR evaluation
#             sample = list()
#             y_true  = list()
#             for head, relation, tail in test_triples:
#                 for candidate_region in args.regions:
#                     y_true.append(1 if candidate_region == tail else 0)
#                     sample.append((head, relation, candidate_region))

#             sample = torch.LongTensor(sample)
#             if args.cuda:
#                 sample = sample.cuda()

#             with torch.no_grad():
#                 y_score = model(sample).squeeze(1).cpu().numpy()

#             y_true = np.array(y_true)

#             #average_precision_score is the same as auc_pr
#             auc_pr = average_precision_score(y_true, y_score)

#             metrics = {'auc_pr': auc_pr}
            
#         else:
#             #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
#             #Prepare dataloader for evaluation
#             if not hasattr(model, 'test_dataset_list'):
#                 test_dataloader_head = DataLoader(
#                     TestDataset(
#                         test_triples, 
#                         all_true_triples, 
#                         args.nentity, 
#                         args.nrelation, 
#                         'head-batch'
#                     ), 
#                     batch_size=args.test_batch_size,
#                     num_workers=max(1, args.cpu_num//2), 
#                     collate_fn=TestDataset.collate_fn_ce,
#                 )

#                 test_dataloader_tail = DataLoader(
#                     TestDataset(
#                         test_triples, 
#                         all_true_triples, 
#                         args.nentity, 
#                         args.nrelation, 
#                         'tail-batch'
#                     ), 
#                     batch_size=args.test_batch_size,
#                     num_workers=max(1, args.cpu_num//2), 
#                     collate_fn=TestDataset.collate_fn_ce,
#                 )
                
#                 test_dataset_list = [[_ for _ in test_dataloader_head], [_ for _ in test_dataloader_tail]]
#                 model.test_dataset_list = test_dataset_list
#             else:
#                 test_dataset_list = model.test_dataset_list
            
#             logs = []

#             step = 0
#             total_steps = sum([len(dataset) for dataset in test_dataset_list])

#             with torch.no_grad():
#                 for test_dataset in test_dataset_list:
#                     for sample, filter_bias, mode in test_dataset:
#                         if args.cuda:
#                             sample = sample.cuda()
#                             filter_bias = filter_bias.cuda()

#                         batch_size = sample.size(0)

#                         scores, rotate_score, identity_matrix_score = model(sample, mode)
#                         scores += filter_bias * 1e4

#                         #Explicitly sort all the entities to ensure that there is no test exposure bias
#                         argsort = torch.argsort(scores, dim = 1, descending=True)

#                         if mode == 'head-batch':
#                             positive_arg = sample[:, 0]
#                         elif mode == 'tail-batch':
#                             positive_arg = sample[:, 2]
#                         else:
#                             raise ValueError('mode %s not supported' % mode)

#                         for i in range(batch_size):
#                             #Notice that argsort is not ranking
#                             ranking = (argsort[i, :] == positive_arg[i]).nonzero()
#                             assert ranking.size(0) == 1

#                             #ranking + 1 is the true ranking used in evaluation metrics
#                             ranking = 1 + ranking.item()
#                             logs.append({
#                                 'MRR': 1.0/ranking,
#                                 'MR': float(ranking),
#                                 'HITS@1': 1.0 if ranking <= 1 else 0.0,
#                                 'HITS@3': 1.0 if ranking <= 3 else 0.0,
#                                 'HITS@10': 1.0 if ranking <= 10 else 0.0,
#                             })

#                         if step % args.test_log_steps == 0:
#                             logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

#                         step += 1

#             metrics = {}
#             for metric in logs[0].keys():
#                 metrics[metric] = sum([log[metric] for log in logs])/len(logs)

#         return metrics