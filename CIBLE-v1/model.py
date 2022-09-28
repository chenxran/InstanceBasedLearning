from functools import partial
from typing import Any, ClassVar, Mapping, Optional, Tuple, cast

import torch
import torch.autograd
from regex import R
from torch import embedding, linalg, nn
from torch.nn import functional

from pykeen.constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from pykeen.losses import CrossEntropyLoss
from pykeen.models.base import EntityRelationEmbeddingModel
from pykeen.models.nbase import ERModel
from pykeen.moves import irfft, rfft
from pykeen.nn import representation_resolver
from pykeen.nn.init import xavier_uniform_, xavier_uniform_norm_, init_phases
from pykeen.nn.representation import Representation, build_representation
from pykeen.typing import (Constrainer, Hint, InductiveMode, Initializer,
                           Normalizer)
from pykeen.utils import clamp_norm, complex_normalize

import copy
from collections import defaultdict
import time


def _projection_initializer(
    x: torch.FloatTensor,
    num_relations: int,
    embedding_dim: int,
    relation_dim: int,
) -> torch.FloatTensor:
    """Initialize by Glorot."""
    return torch.nn.init.xavier_uniform_(x.view(num_relations, embedding_dim, relation_dim)).view(x.shape)

# activation function
ACTFN = {
    None: lambda x: x,
    "none": lambda x: x,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "elu": functional.elu,
    "selu": torch.selu,
    "leaky_relu": functional.leaky_relu,
    "gelu": functional.gelu,
}


def construct_h_list(cls):
    cls.h_list_new = [{"h": [], "t": []} for _ in range(cls.num_relations)]
    for h, r, t in cls.triples_factory._add_inverse_triples_if_necessary(cls.triples_factory.mapped_triples):
        h, r, t = h.item(), r.item(), t.item()
        cls.h_list_new[r]["h"].append(h)
        cls.h_list_new[r]["t"].append(t)
    
    for key in range(cls.num_relations):
        cls.h_list_new[key]["h"] = torch.tensor(cls.h_list_new[key]["h"]).cuda()
        cls.h_list_new[key]["t"] = torch.tensor(cls.h_list_new[key]["t"]).cuda()


def scoring(cls, h, r):
    # calculate identity matrix
    identity_matrix = cls.identity_matrix(h, r)  # [batch_size, num_entities] for h and h'

    # construct indice
    batch_indices, candidate_t_indices, h_indices = list(), list(), list()
    for i, _r in enumerate(r):
        dictionary = cls.h_list_new[_r]
        candidate_t_indices.append(dictionary["t"])
        h_indices.append(dictionary["h"])

        length =len(dictionary["t"])
        batch_indices.append(torch.full(size=(length, ), fill_value=i, dtype=torch.long, device=h.device))

    candidate_t_indices, h_indices, batch_indices = torch.cat(candidate_t_indices), torch.cat(h_indices), torch.cat(batch_indices)

    # given (h, r), calculate scores for all candidate t
    # for each t_i where i from 0 to |E| - 1, we first find all h' that can arrive t_i via r.
    # Then, we sum up the scores of Identity_Matrix(h, h', r)
    selected_score = identity_matrix[batch_indices.long(), h_indices.long()]  # len(batch_indices)
    # selected_score[i] -> scores[batch_indices[i], candidate_t_indices[i], h_indices[i]]
    scores = torch.sparse.sum(
        torch.sparse_coo_tensor(
            values=selected_score,
            indices=torch.vstack([batch_indices, candidate_t_indices, h_indices]),
            size=(len(h), cls.triples_factory.num_entities, cls.triples_factory.num_entities),
        ),
        dim=2,
    ).to_dense()

    if cls.args.mlp:
        scores = cls.mlp(scores)
    return scores


class MFModel(EntityRelationEmbeddingModel):
    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=dict(type=int, low=128, high=1024, q=128),
        activation=dict(type="categorical", choices=ACTFN.keys()),
    )

    def __init__(
        self,
        *,
        args,
        triples_factory,
        embedding_dim: int = 50,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = functional.normalize,
        relation_initializer: Hint[Initializer] = xavier_uniform_norm_,
        relation_constrainer: Hint[Constrainer] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                constrainer=entity_constrainer,
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
                constrainer=relation_constrainer,
            ),
            **kwargs,
        )
        self.args = args
        self.triples_factory = triples_factory
        self.activation = args.activation
        self.selfmask = args.selfmask
        construct_h_list(self)

        self.mf_entity_embeddings = self.entity_embeddings
        self.mf_relation_embeddings = self.relation_embeddings

        if self.args.diag_w:
            self.diag_w = nn.Parameter(torch.ones(embedding_dim))

        if self.args.w:
            self.w = nn.Linear(embedding_dim, embedding_dim)
            weight = torch.zeros(embedding_dim, embedding_dim)
            weight[torch.arange(embedding_dim), torch.arange(embedding_dim)] = 1
            self.w.weight.data = weight.float()
            self.w.bias.data = torch.zeros(embedding_dim).float()

    def score_hrt(self, hrt_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        raise NotImplementedError

    def score_t(self, hr_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        scores = scoring(self, hr_batch[:, 0], hr_batch[:, 1])
        return scores

    def identity_matrix(self, h, r):
        # prepare identity_matrix
        e1, e2, rel = self.mf_entity_embeddings(h), self.mf_entity_embeddings(indices=None), self.mf_relation_embeddings(r)     
        if self.args.diag_w:
            rel = rel * self.diag_w
        elif self.args.w:
            rel = self.w(rel)

        tmp = e1.unsqueeze(1) * e2.unsqueeze(0)
        if self.args.normalize:
            identity_matrix = torch.bmm(tmp / tmp.norm(dim=2).unsqueeze(2), rel.unsqueeze(2))  # [batch_size, num_entities] 
        else:
            identity_matrix = torch.bmm(tmp, rel.unsqueeze(2))  # [batch_size, num_entities] 

        if self.selfmask:
            identity_matrix[torch.arange(h.size(0)), h, :] = 0
        identity_matrix = ACTFN[self.activation](identity_matrix).reshape(len(h), self.triples_factory.num_entities)   # [batch_size, num_entities]
        return identity_matrix


class MFV3Model(EntityRelationEmbeddingModel):
    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=dict(type=int, low=128, high=1024, q=128),
        activation=dict(type="categorical", choices=ACTFN.keys()),
    )

    def __init__(
        self,
        *,
        args,
        triples_factory,
        embedding_dim: int = 50,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = functional.normalize,
        relation_initializer: Hint[Initializer] = xavier_uniform_norm_,
        relation_constrainer: Hint[Constrainer] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                constrainer=entity_constrainer,
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
                constrainer=relation_constrainer,
            ),
            **kwargs,
        )
        self.args = args
        self.triples_factory = triples_factory
        self.activation = args.activation
        self.selfmask = args.selfmask
        self.scoring_fct_norm = args.scoring_fct_norm

        construct_h_list(self)

        self.mf_entity_embeddings = self.entity_embeddings
        self.mf_relation_embeddings = self.relation_embeddings

        if self.args.mlp:
            if self.args.intermediate_dim is None:
                self.intermediate_dim = 2 * self.triples_factory.num_entities
            else:
                self.intermediate_dim = self.args.intermediate_dim
            self.mlp = nn.Sequential(
                nn.Linear(self.triples_factory.num_entities, self.intermediate_dim),
                nn.Tanh(),
                nn.Dropout(p=0.1),
                nn.Linear(self.intermediate_dim, self.triples_factory.num_entities),
            )

    def score_hrt(self, hrt_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        raise NotImplementedError

    def score_t(self, hr_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        scores = scoring(self, hr_batch[:, 0], hr_batch[:, 1])
        return scores

    def identity_matrix(self, h, r):
        # prepare identity_matrix
        e1, e2 = self.mf_entity_embeddings(h), self.mf_entity_embeddings(indices=None)

        if self.args.normalize:
            e1 = clamp_norm(e1, p=self.scoring_fct_norm, dim=-1, maxnorm=1.0)
            e2 = clamp_norm(e2, p=self.scoring_fct_norm, dim=-1, maxnorm=1.0)    
        identity_matrix = torch.matmul(e1, e2.t())

        if self.selfmask:
            identity_matrix[torch.arange(h.size(0)), h, :] = 0
        identity_matrix = ACTFN[self.activation](identity_matrix).reshape(len(h), self.triples_factory.num_entities)   # [batch_size, num_entities]
        return identity_matrix



class CIBLETransEModel(MFV3Model):
    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=dict(type=int, low=128, high=1024, q=128),
        activation=dict(type="categorical", choices=ACTFN.keys()),
    )

    def __init__(
        self,
        *,
        args,
        triples_factory,
        embedding_dim: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(
            args=args,
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            **kwargs,
        )

        if isinstance(args.transe_weight, float):
            self.transe_weight = args.transe_weight
            self.bias = 0.
        elif args.transe_weight == "trainable":
            self.transe_weight = nn.Parameter(torch.randn(1))
            self.bias = nn.Parameter(torch.randn(1))
        else:
            raise NotImplementedError

        self.scoring_fct_norm = args.scoring_fct_norm

        if self.args.not_share_entity_embedding:
            self.mf_entity_embeddings = copy.deepcopy(self.entity_embeddings)

        if self.args.not_share_relation_embedding:
            self.mf_relation_embeddings = copy.deepcopy(self.relation_embeddings)

    def transe_score(self, h, r, t=None):
        h, r = self.entity_embeddings(h), self.relation_embeddings(r)
        t = self.entity_embeddings(indices=None)
        return - linalg.vector_norm(h[:, None, :] + r[:, None, :] - t[None, :, :], dim=-1, ord=self.scoring_fct_norm)

    def score_t(self, hr_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        scores = scoring(self, hr_batch[:, 0], hr_batch[:, 1])
        return scores + self.transe_weight * self.transe_score(hr_batch[:, 0], hr_batch[:, 1]) + self.bias

class CIBLEDistMultModel(MFModel):
    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=dict(type=int, low=128, high=1024, q=128),
        activation=dict(type="categorical", choices=ACTFN.keys()),
    )

    def __init__(
        self,
        *,
        args,
        triples_factory,
        embedding_dim: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(
            args=args,
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            **kwargs,
        )

        if isinstance(args.transe_weight, float):
            self.transe_weight = args.transe_weight
        elif args.transe_weight == "trainable":
            self.transe_weight = nn.Parameter(torch.randn(1))
        else:
            raise NotImplementedError

    def distmult_score(self, h, r, t=None):
        # Get embeddings
        h = self.entity_embeddings(indices=h).view(-1, 1, self.embedding_dim)
        r = self.relation_embeddings(indices=r).view(-1, 1, self.embedding_dim)
        t = self.entity_embeddings(indices=None).view(1, -1, self.embedding_dim)

        # Rank against all entities
        return torch.sum(h * r * t, dim=-1)

    def score_t(self, hr_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        scores = scoring(self, hr_batch[:, 0], hr_batch[:, 1])
        return scores + self.transe_weight * self.distmult_score(hr_batch[:, 0], hr_batch[:, 1])

class MFV2Model(EntityRelationEmbeddingModel):
    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        relation_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        *,
        args,
        triples_factory,
        entity_embedding_dim: int = 50,
        relation_embedding_dim: int = 30,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = clamp_norm,  # type: ignore
        relation_initializer: Hint[Initializer] = xavier_uniform_norm_,
        relation_constrainer: Hint[Constrainer] = clamp_norm,  # type: ignore
        **kwargs,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            entity_representations_kwargs=dict(
                shape=entity_embedding_dim,
                initializer=entity_initializer,
                constrainer=entity_constrainer,
                constrainer_kwargs=dict(maxnorm=1.0, p=2, dim=-1),
            ),
            relation_representations_kwargs=dict(
                shape=relation_embedding_dim,
                initializer=relation_initializer,
                constrainer=relation_constrainer,
                constrainer_kwargs=dict(maxnorm=1.0, p=2, dim=-1),
            ),
            **kwargs,
        )
        self.args = args
        self.triples_factory = triples_factory
        self.activation = args.activation
        self.selfmask = args.selfmask
        self.scoring_fct_norm = args.scoring_fct_norm
        
        self.construct_h_list()

        # embeddings
        self.relation_projections = representation_resolver.make(
            query=None,
            shape=(relation_embedding_dim * entity_embedding_dim,),
            max_id=self.num_relations,
            initializer=partial(
                _projection_initializer,
                num_relations=self.num_relations,
                embedding_dim=self.embedding_dim,
                relation_dim=self.relation_dim,
            ),
        )

        self.mf_entity_embeddings = self.entity_embeddings
        self.mf_relation_embeddings = self.relation_embeddings
        self.mf_relation_projections = self.relation_projections

        if self.args.mlp:
            if self.args.intermediate_dim is None:
                self.intermediate_dim = 2 * self.triples_factory.num_entities
            else:
                self.intermediate_dim = self.args.intermediate_dim
            self.mlp = nn.Sequential(
                nn.Linear(self.triples_factory.num_entities, self.intermediate_dim),
                nn.Tanh(),
                nn.Dropout(p=0.1),
                nn.Linear(self.intermediate_dim, self.triples_factory.num_entities),
            )

    def construct_h_list(self):
        self.h_list_new = [{"h": [], "t": []} for _ in range(self.num_relations)]
        for h, r, t in self.triples_factory._add_inverse_triples_if_necessary(self.triples_factory.mapped_triples):
            h, r, t = h.item(), r.item(), t.item()
            self.h_list_new[r]["h"].append(h)
            self.h_list_new[r]["t"].append(t)
        
        for key in range(self.num_relations):
            self.h_list_new[key]["h"] = torch.tensor(self.h_list_new[key]["h"]).cuda()
            self.h_list_new[key]["t"] = torch.tensor(self.h_list_new[key]["t"]).cuda()

        self.h_list_new_2 = defaultdict(list)
        for h, r, t in self.triples_factory._add_inverse_triples_if_necessary(self.triples_factory.mapped_triples):
            h, r, t = h.item(), r.item(), t.item()
            self.h_list_new_2[(r, t)].append(h)

        for k, v in self.h_list_new_2.items():
            self.h_list_new_2[k] = torch.tensor(v).cuda()

    def score_hrt(self, hrt_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        raise NotImplementedError

    def score_t(self, hr_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        scores = self.scoring(hr_batch[:, 0], hr_batch[:, 1])
        return scores

    def identity_matrix(self, h, r, t=None):
        # construct indice
        batch_indices, candidate_t_indices, h_indices = list(), list(), list()
        for i, _r in enumerate(r):
            dictionary = self.h_list_new[_r]
            candidate_t_indices.append(dictionary["t"])
            h_indices.append(dictionary["h"])

            length = len(dictionary["t"])
            batch_indices.append(torch.full(size=(length, ), fill_value=i, dtype=torch.long, device=h.device))

        candidate_t_indices, h_indices, batch_indices = torch.cat(candidate_t_indices), torch.cat(h_indices), torch.cat(batch_indices)

        # prepare identity_matrix
        e1 = self.mf_entity_embeddings(h).unsqueeze(dim=1)
        e2 = self.mf_entity_embeddings(indices=None).unsqueeze(dim=0)
        rel_proj = self.mf_relation_projections(indices=r).view(-1, self.embedding_dim, self.relation_dim)

        e1_bot = e1 @ rel_proj
        e2_bot = e2 @ rel_proj

        if self.args.normalize:
            # ensure constraints
            e1_bot = clamp_norm(e1_bot, p=self.scoring_fct_norm, dim=-1, maxnorm=1.0)
            e2_bot = clamp_norm(e2_bot, p=self.scoring_fct_norm, dim=-1, maxnorm=1.0)

        identity_matrix = torch.bmm(e1_bot, e2_bot.permute(0, 2, 1).contiguous()).squeeze()

        if self.selfmask:
            identity_matrix[torch.arange(h.size(0)), h, :] = 0
        identity_matrix = ACTFN[self.activation](identity_matrix).reshape(len(h), self.triples_factory.num_entities)   # [batch_size, num_entities]
        return identity_matrix, candidate_t_indices, h_indices, batch_indices

    def scoring(self, h, r):
        # calculate identity matrix
        identity_matrix, candidate_t_indices, h_indices, batch_indices = self.identity_matrix(h, r)  # [batch_size, num_entities] for h and h'

        # given (h, r), calculate scores for all candidate t
        # for each t_i where i from 0 to |E| - 1, we first find all h' that can arrive t_i via r.
        # Then, we sum up the scores of Identity_Matrix(h, h', r)
        selected_score = identity_matrix[batch_indices.long(), h_indices.long()]  # len(batch_indices)

        # selected_score[i] -> scores[batch_indices[i], candidate_t_indices[i], h_indices[i]]
        sparse_selected_scores = torch.sparse_coo_tensor(
                values=selected_score,
                indices=torch.vstack([batch_indices, candidate_t_indices, h_indices]),
                size=(len(h), self.triples_factory.num_entities, self.triples_factory.num_entities),
        )

        scores = torch.sparse.sum(sparse_selected_scores, dim=2).to_dense()
        
        if self.args.mlp:
            scores = self.mlp(scores)
        return scores

    def scoring2(self, h, r):
        batch_size = len(h)
        
        prototypes = []
        scores = torch.cuda.FloatTensor(batch_size, self.num_entities) * 0.  # [batch_size, num_entities]

        for i in range(batch_size):
            if r[i] % 2 == 0:
                temp = self.h_list_new_2[(r[i].item() + 1, h[i].item())]
            else:
                temp = self.h_list_new_2[(r[i].item() - 1, h[i].item())]
            if len(temp) != 0:
                prototypes = temp
                e1 = self.mf_entity_embeddings(indices=None).unsqueeze(dim=0)
                e2 = self.mf_entity_embeddings(indices=prototypes).unsqueeze(dim=0)
                if r[i] % 2 == 0:
                    rel_proj = self.mf_relation_projections(indices=r[i] + 1).view(-1, self.embedding_dim, self.relation_dim)
                else:
                    rel_proj = self.mf_relation_projections(indices=r[i] - 1).view(-1, self.embedding_dim, self.relation_dim)

                e1_bot = e1 @ rel_proj  # [num_entities, relation_dim]
                e2_bot = e2 @ rel_proj  # [len(prototypes), relation_dim]        

                if self.args.normalize:
                    # ensure constraints
                    e1_bot = clamp_norm(e1_bot, p=self.scoring_fct_norm, dim=-1, maxnorm=1.0)
                    e2_bot = clamp_norm(e2_bot, p=self.scoring_fct_norm, dim=-1, maxnorm=1.0)

                # identity_matrix = e1_bot @ e2_bot.t()
                identity_matrix = torch.bmm(e1_bot, e2_bot.permute(0, 2, 1).contiguous())  # [len(indices), num_entities, prototypes]

                if self.selfmask:
                    identity_matrix[torch.arange(h.size(0)), h, :] = 0
                identity_matrix = ACTFN[self.activation](identity_matrix).reshape(1, self.triples_factory.num_entities, len(prototypes))   # [1, num_entities, prototypes]

                scores = torch.cuda.FloatTensor(batch_size, self.num_entities) * 0.  # [batch_size, num_entities]
                scores[i, :] = identity_matrix.sum(dim=2)
            else:
                continue
        
        return scores

class CIBLETransR(MFV2Model):
    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        relation_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        *,
        args,
        triples_factory,
        entity_embedding_dim: int = 50,
        relation_embedding_dim: int = 30,
        **kwargs,
    ) -> None:
        super().__init__(
            args=args,
            triples_factory=triples_factory,
            entity_embedding_dim=entity_embedding_dim,
            relation_embedding_dim=relation_embedding_dim,
            **kwargs,
        )

        if isinstance(args.transe_weight, float):
            self.transe_weight = args.transe_weight
        elif args.transe_weight == "trainable":
            self.transe_weight = nn.Parameter(torch.randn(1))

        if self.args.not_share_entity_embedding:
            self.mf_entity_embeddings = copy.deepcopy(self.entity_embeddings)

        if self.args.not_share_relation_embedding:
            self.mf_relation_embeddings = copy.deepcopy(self.relation_embeddings)

    def transr_score(self, h, r, t=None):
        h = self.entity_embeddings(indices=h).unsqueeze(dim=1)
        m_r = self.relation_projections(indices=r).view(-1, self.embedding_dim, self.relation_dim)
        r = self.relation_embeddings(indices=r).unsqueeze(dim=1)
        t = self.entity_embeddings(indices=None).unsqueeze(dim=0)

        h_bot = h @ m_r
        t_bot = t @ m_r
        # ensure constraints
        h_bot = clamp_norm(h_bot, p=self.scoring_fct_norm, dim=-1, maxnorm=1.0)
        t_bot = clamp_norm(t_bot, p=self.scoring_fct_norm, dim=-1, maxnorm=1.0)

        # evaluate score function, shape: (b, e)
        return self.args.bias - linalg.vector_norm(h_bot + r - t_bot, ord=self.scoring_fct_norm, dim=-1)

    def score_t(self, hr_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:
        # This is used to run TransR baseline under the same hyper-parameter setting.
        if not self.args.no_identity_matrix:
            scores = self.scoring(hr_batch[:, 0], hr_batch[:, 1])
            return scores + self.transe_weight * self.transr_score(hr_batch[:, 0], hr_batch[:, 1])
        else:
            return self.transr_score(hr_batch[:, 0], hr_batch[:, 1])

