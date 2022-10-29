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
import math
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)


ACTFN = {
    'none': lambda x: x,
    'tanh': torch.tanh,
}


class IBLE(nn.Module):
    def __init__(self, args, edges, nrelation, nentity, gamma, mlp=False, relation_aware=False, entity_dim=None):
        super().__init__()
        self.config = args
        self.use_mlp = mlp
        self.relation_aware = relation_aware
        self.m = len(edges)
        self.n = nentity
        self.node_edge1 = []
        self.edge_node1 = []
        self.r_mask1 = torch.zeros(nrelation, self.m).cuda()
        self.r_mask2 = torch.zeros(nrelation, self.m).cuda()
        self.node_edge2 = []
        self.edge_node2 = []
        self.gamma = gamma

        for i, (h,r,t) in enumerate(edges):
            self.node_edge1.append([h, i])
            self.edge_node1.append([i, t])
            self.r_mask1[r][i] = 1
            self.r_mask2[r][i] = 1

            self.node_edge2.append([t, i])
            self.edge_node2.append([i, h])

        self.node_edge1 = torch.LongTensor(self.node_edge1).permute(1, 0).cuda()
        self.edge_node1 = torch.LongTensor(self.edge_node1).permute(1, 0).cuda()
        self.node_edge2 = torch.LongTensor(self.node_edge2).permute(1, 0).cuda()
        self.edge_node2 = torch.LongTensor(self.edge_node2).permute(1, 0).cuda()

        self.aggregator1 = MessagePassing(aggr='add')
        self.aggregator2 = MessagePassing(aggr=self.config.pooling)

        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(self.n, min(2 * self.n, 512)),
                nn.Tanh(),
                nn.Dropout(p=0.1),
                nn.Linear(min(2 * self.n, 512), self.n),
            )

        hidden_dim = entity_dim * 2 if self.config.double_entity_embedding else entity_dim
        if self.relation_aware == 'diag':
            self.head_r = nn.Parameter(torch.randn(nrelation, hidden_dim))
            self.tail_r = nn.Parameter(torch.randn(nrelation, hidden_dim))
        elif self.relation_aware == 'matrix':
            self.head_r = nn.Parameter(torch.randn(nrelation, hidden_dim, hidden_dim))
            self.tail_r = nn.Parameter(torch.randn(nrelation, hidden_dim, hidden_dim))

        if self.relation_aware is not None:
            bound = 1.0 * 6 / math.sqrt(hidden_dim)
            torch.nn.init.uniform_(self.head_r, -bound, bound)
            torch.nn.init.uniform_(self.tail_r, -bound, bound)

    def forward(self, emb, all_emb, source_idx, relation_ids, mode): # bs * dim, nentity * dim
        bs = emb.size(0)
        if mode == 'head-batch':
            node_edge = self.node_edge2
            edge_node = self.edge_node2
            r_mask = self.r_mask2
        else:
            node_edge = self.node_edge1
            edge_node = self.edge_node1
            r_mask = self.r_mask1

        if self.relation_aware == 'diag':
            if mode == 'head-batch':
                rel = self.head_r[relation_ids]
            else:
                rel = self.tail_r[relation_ids]
            emb = (emb * rel).unsqueeze(0)
            all_emb = all_emb.unsqueeze(1) * rel.unsqueeze(0)
        elif self.relation_aware == 'matrix':
            if mode == 'head-batch':
                rel = self.head_r[relation_ids]
            else:
                rel = self.tail_r[relation_ids]
            emb = torch.bmm(emb.unsqueeze(1), rel)
            all_emb = torch.bmm(all_emb.repeat(bs, 1, 1), rel)
        elif self.relation_aware is None:
            emb = emb.unsqueeze(0)
            all_emb = all_emb.unsqueeze(1)

        if self.config.cosine:
            if self.relation_aware == 'diag':
                dis = ACTFN[self.config.activation](torch.bmm(all_emb.permute(1, 0, 2).contiguous(), emb.permute(1, 2, 0).contiguous()).squeeze(2).t())
            elif self.relation_aware is None:
                dis = ACTFN[self.config.activation](torch.matmul(all_emb, emb.permute(0, 2, 1).contiguous()).squeeze(1))
            elif self.relation_aware == 'matrix':
                dis = ACTFN[self.config.activation](torch.matmul(all_emb, emb.permute(0, 2, 1).contiguous()).squeeze(2).t())
        else:
            dis = (emb - all_emb).norm(p=1, dim=-1)

            if source_idx is not None:
                self_mask = torch.ones(self.n, bs).bool().cuda()
                self_mask[source_idx, torch.arange(bs)] = False
                dis = torch.where(self_mask,dis,torch.tensor(1e8).cuda())
            dis = torch.sigmoid(self.gamma-dis)

        edge_score = self.aggregator1.propagate(node_edge,x=dis, size = (self.n,self.m)) #m * bs
        edge_score *= torch.index_select(r_mask, dim=0, index=relation_ids).permute(1,0) #m*bs
        target_score = self.aggregator2.propagate(edge_node,x = edge_score, size = (self.m,self.n) ).permute(1,0) #bs * n

        if self.use_mlp:
            target_score = self.mlp(target_score)
        return target_score

class KGEModel(nn.Module):
    def __init__(self, args, model_name, nentity, nrelation, hidden_dim, gamma, mlp=False, relation_aware=False,
                 double_entity_embedding=False, double_relation_embedding=False, train_triples = None, ible_weight = 0.0):
        super(KGEModel, self).__init__()
        self.config = args
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.ible_weight = ible_weight
        if ible_weight!= 0.0:
            self.ible = IBLE(args, train_triples, nrelation, nentity, gamma, mlp=mlp, relation_aware=relation_aware, entity_dim=hidden_dim)

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        
        bound = 1.0 * 6 / math.sqrt(self.entity_embedding.shape[-1])
        torch.nn.init.uniform_(self.entity_embedding, -bound, bound)

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))

        bound = 1.0 * 6 / math.sqrt(self.relation_embedding.shape[-1])
        torch.nn.init.uniform_(self.relation_embedding, -bound, bound)
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
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
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
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
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
     
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }
        
        if self.model_name in model_func:
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
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

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
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        for _ in range(args.gradient_accumulation_steps):
            positive_sample, negative_sample, subsampling_weight, mode, label = next(train_iterator)   # this label is deprecated

            if args.cuda:
                positive_sample = positive_sample.cuda()
                negative_sample = negative_sample.cuda()
                subsampling_weight = subsampling_weight.cuda()

            if model.config.negative_sample_size == -1:
                
                if 0.0 < model.ible_weight < 1.0:
                    negative_score = model((positive_sample, negative_sample), mode=mode)
                    source_idx = positive_sample[:,2] if mode == 'head-batch' else positive_sample[:,0]

                    emb_batch = torch.index_select(model.entity_embedding,dim=0,index=source_idx)
                    emb_all = model.entity_embedding

                    ible_score = model.ible(emb_batch,emb_all,source_idx,positive_sample[:,1],mode)
                    negative_score = model.merge_score(negative_score,ible_score)
                elif model.ible_weight == 0.0:
                    negative_score = model((positive_sample, negative_sample), mode=mode)
                elif model.ible_weight == 1.0:
                    source_idx = positive_sample[:,2] if mode == 'head-batch' else positive_sample[:,0]

                    emb_batch = torch.index_select(model.entity_embedding,dim=0,index=source_idx)
                    emb_all = model.entity_embedding

                    negative_score = model.ible(emb_batch,emb_all,source_idx,positive_sample[:,1],mode)

                labels = positive_sample[:,0] if mode == 'head-batch' else positive_sample[:,2]
                loss = F.cross_entropy(negative_score, labels)
            else:
                if 0.0 < model.ible_weight < 1.0:
                    negative_score = model((positive_sample, negative_sample), mode=mode)
                    positive_score = model(positive_sample)

                    source_idx = positive_sample[:,2] if mode == 'head-batch' else positive_sample[:,0]

                    emb_batch = torch.index_select(model.entity_embedding,dim=0,index=source_idx)
                    emb_all = model.entity_embedding
                    ible_score = model.ible(emb_batch, emb_all, source_idx, positive_sample[:,1], mode)
                    
                    if mode == 'head-batch':
                        positive_ible_score = ible_score[torch.arange(positive_sample.shape[0]), positive_sample[:,0]]
                    else:
                        positive_ible_score = ible_score[torch.arange(positive_sample.shape[0]), positive_sample[:,2]]
                    positive_score = model.merge_score(positive_score, positive_ible_score.unsqueeze(1))
                    
                    negative_ible_score = torch.vstack([ible_score[i, negative_sample[i]] for i in range(negative_sample.shape[0])])
                    negative_score = model.merge_score(negative_score, negative_ible_score)
                elif model.ible_weight == 0.0:
                    negative_score = model((positive_sample, negative_sample), mode=mode)
                    positive_score = model(positive_sample)
                elif model.ible_weight == 1.0:
                    source_idx = positive_sample[:,2] if mode == 'head-batch' else positive_sample[:,0]

                    emb_batch = torch.index_select(model.entity_embedding,dim=0,index=source_idx)
                    emb_all = model.entity_embedding
                    ible_score = model.ible(emb_batch, emb_all, source_idx, positive_sample[:,1], mode)
                    
                    if mode == 'head-batch':
                        positive_score = ible_score[torch.arange(positive_sample.shape[0]), positive_sample[:,0]]
                    else:
                        positive_score = ible_score[torch.arange(positive_sample.shape[0]), positive_sample[:,2]]
                    negative_score = torch.vstack([ible_score[i, negative_sample[i]] for i in range(negative_sample.shape[0])])

                if model.config.loss == 'crossentropy':
                    scores = torch.cat([positive_score, negative_score], dim=1)
                    labels = torch.zeros(scores.size(0), device=scores.device).long()

                    loss = F.cross_entropy(scores, target=labels)

                elif model.config.loss == 'margin':
                    if args.negative_adversarial_sampling:
                        #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                                        * F.logsigmoid(-negative_score)).sum(dim = 1)
                    else:
                        negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

                    positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

                    if args.uni_weight:
                        positive_sample_loss = - positive_score.mean()
                        negative_sample_loss = - negative_score.mean()
                    else:
                        positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
                        negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

                    loss = (positive_sample_loss + negative_sample_loss) / 2


                    # positive_sample_loss = - positive_score.mean()
                    # negative_sample_loss = negative_score.mean()

                    # loss = (positive_sample_loss + negative_sample_loss) / 2

            #loss = loss_fn(negative_score, labels)

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

        log = {
            **regularization_log,
            # 'positive_sample_loss': positive_sample_loss.item(),
            # 'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    def merge_score(self, origin_score, ible_score):
        return ible_score * self.ible_weight + torch.sigmoid(origin_score) * (1 - self.ible_weight)

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

                        score = model((positive_sample, negative_sample), mode)
                        if model.ible_weight != 0.0:
                            if mode == 'head-batch':
                                source_idx = positive_sample[:, 2]
                            else:
                                source_idx = positive_sample[:, 0]
                            emb_batch = torch.index_select(model.entity_embedding, dim=0, index=source_idx)
                            emb_all = model.entity_embedding
                            ible_score = model.ible(emb_batch, emb_all, source_idx, positive_sample[:,1], mode)
                            score = model.merge_score(score,ible_score)

                        score += filter_bias * 10000

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
