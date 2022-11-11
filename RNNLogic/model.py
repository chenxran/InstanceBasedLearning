import copy
import gc
from collections import defaultdict
from tempfile import tempdir

import torch
from torch import nn
from transformers import BertForSequenceClassification
import groundings
from knowledge_graph_utils import mask2list, list2mask, build_graph
from metrics import Metrics
from reasoning_model import ReasoningModel
from rotate import RotatE
from generator import RNNLogicGenerator, BartLogicGenerator
from discriminator import Discriminator
from path_scorer import PathScorer
from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np

class RNNLogic(object):
    def __init__(self, dataset, args, print=print):
        super(RNNLogic, self).__init__()
        self.E = dataset['E']
        self.R = dataset['R']
        assert self.R % 2 == 0
        self.set_args(vars(args))

        self.dataset = dataset
        self._print = print
        self.print = self.log_print
        self.predictor_init = lambda: LogicPredictor(self.dataset, args, print=self.log_print)

        print = self.print
        print("RNNLogic Init", self.E, self.R)

    def log_print(self, *args, **kwargs):
        import datetime
        timestr = datetime.datetime.now().strftime("%H:%M:%S.%f")
        if hasattr(self, 'em'):
            emstr = self.em if self.em < self.num_em_epoch else '#'
            prefix = f"r = {self.r} EM = {emstr}"
        else:
            prefix = "init"
        self._print(f"[{timestr}] {prefix} | ", end="")
        self._print(*args, **kwargs)

    # Use EM algorithm to train RNNLogic model
    def train_model(self, r, rule_file=None, model_file=None):
        if rule_file is None:
            rule_file = f"rules_{r}.txt"
        if model_file is None:
            model_file = f"model_{r}.pth"
        num_em_epoch = self.arg('num_em_epoch')

        self.num_em_epoch = num_em_epoch
        self.r = r
        print = self.print

        pgnd_buffer = dict()
        rgnd_buffer = dict()
        rgnd_buffer_test = dict()

        self.em = 0
        # def generate_rules():
            # since we have not trained the generator before, we random sample 1000 rules and initialize predictor to find the high-quality rules
            # if self.em == 0:
            # else:
            #     sampled = set()
            #     sampled.add((r,))
            #     sampled.add(tuple())

            #     rules = [(r,)]
            #     prior = [0.0, ]
            #     for rule, score in self.generator.generate_rules(r,
            #                                                   self.arg('max_beam_rules'),
            #                                                   self.predictor.arg('max_rule_len')):
            #         rule = tuple(rule)
            #         if rule in sampled:
            #             continue
            #         sampled.add(rule)
            #         rules.append(rule)
            #         prior.append(score)
            #         if len(sampled) % self.arg('sample_print_epoch') == 0:
            #             print(f"sampled # = {len(sampled)} rule = {rule} score = {score}")

            #     print(f"Done |sampled| = {len(sampled)}")

            #     prior = torch.tensor(prior).cuda()
            #     prior -= prior.max()
            #     prior = prior.exp()

            #     self.predictor.relation_init(r, rules=rules, prior=prior)


        # for self.em in range(num_em_epoch):
        self.predictor = self.predictor_init()
        self.predictor.pgnd_buffer = pgnd_buffer
        self.predictor.rgnd_buffer = rgnd_buffer
        self.predictor.rgnd_buffer_test = rgnd_buffer_test

        # generate rule for each example
        self.predictor.relation_init(r=r, rule_file=rule_file, force_init_weight=self.arg('init_weight_boot'))
        # generate_rules()
        valid, test = self.predictor.train_model()

        # is_ibl_rules = []
        # num_can = []
        # for k, v in self.predictor.rgnd_buffer.items():
        #     is_ibl_rules.append(v.is_ibl_rule)
        #     num_can.append(len(v))
        # return num_can

        ckpt = {
            'r': r,
            'metrics': {
                'valid': valid,
                'test': test
            },
            'args': self._args_init,
            'rules': self.predictor.rules_exp,
            'predictor': self.predictor.state_dict(),
        }
        torch.save(ckpt, model_file)
        gc.collect()

        return valid, test

    def arg(self, name, apply=None):
        v = self._args[name]
        if apply is None:
            if v is None:
                return None
            return eval(v)
        return apply(v)

    # Definitions for EM framework
    def set_args(self, args):
        self._args_init = args
        self._args = dict()

        def make_str(v):
            if isinstance(v, int):
                return True
            if isinstance(v, float):
                return True
            if isinstance(v, bool):
                return True
            if isinstance(v, str):
                return True
            return False

        for k, v in args.items():
            self._args[k] = str(v) if make_str(v) else v


class LogicPredictor(ReasoningModel):
    def __init__(self, dataset, args, print=print):
        super(LogicPredictor, self).__init__()
        self.E = dataset['E']
        self.R = dataset['R'] + 1
        assert self.R % 2 == 1
        self.dataset = dataset
        self.args = args
        self.set_args(vars(args))
        rotate_pretrained = self.arg('rotate_pretrained', apply=lambda x: x)
        self.rotate = RotatE(dataset, rotate_pretrained)
        self.training = True

        self.rule_weight_raw = torch.nn.Parameter(torch.zeros(1))
        if rotate_pretrained is not None:
            if self.arg('param_relation_embed'):
                self.rotate.enable_parameter('relation_embed')
            if self.arg('param_entity_embed'):
                self.rotate.enable_parameter('entity_embed')

        self.training = True
        self.pgnd_buffer = dict()
        self.rgnd_buffer = dict()
        self.rgnd_buffer_test = dict()
        self.rotate_weight_buffer = dict()
        self.cuda()
        self.print = print
        self.debug = False
        self.recording = False

    def train(self, mode=True):
        self.training = mode
        super(LogicPredictor, self).train(mode)

    def eval(self):
        self.train(False)

    def index_select(self, tensor, index):
        if self.training:
            if not isinstance(index, torch.Tensor):
                index = torch.tensor(index)
            index = index.to(tensor.device)
            return tensor.index_select(0, index).squeeze(0)
        else:
            return tensor[index]

    @staticmethod
    def load_batch(batch):
        return tuple(map(lambda x: x.cuda() if isinstance(x, torch.Tensor) else x, batch))

    # Fetch rule embeddings, either from buffer or by re-calculating
    def rule_embed(self, force=False):
        if not force and not self.arg('param_relation_embed'):
            return self._rule_embed

        relation_embed = self.rotate._attatch_empty_relation()
        # rule_embed = torch.zeros(self.num_rule, self.rotate.embed_dim).cuda()
        rule_embed = torch.cuda.FloatTensor(self.num_rule, self.rotate.embed_dim) * 0.

        # if self.arg("rotate_with_r"):
        # rule_embed += self.index_select(relation_embed, [self.r] * self.rules.size(1))
        # else:
        for i in range(self.MAX_RULE_LEN):
            rule_embed += self.index_select(relation_embed, self.rules[i])
        return rule_embed

    # Init rules
    def set_rules(self, rules: List[Tuple[int]]):
        paths = rules
        r = self.r
        self.eval()

        # self.MAX_RULE_LEN = 0
        # for path in rules:
        # 	self.MAX_RULE_LEN = max(self.MAX_RULE_LEN, len(path))
        self.MAX_RULE_LEN = self.arg('max_rule_len')

        pad = self.R - 1
        gen_end = pad
        gen_pad = self.R
        rules = []
        rules_gen = []
        rules_exp = []

        for path in paths:
            npad = (self.MAX_RULE_LEN - len(path))
            rules.append(path + (pad,) * npad)
            rules_gen.append((r,) + path + (gen_end,) + (gen_pad,) * npad)
            rules_exp.append(tuple(path))

        self.rules = torch.LongTensor(rules).t().cuda()  # rules used for training predictor
        # print(self.rules.size())
        self.rules_gen = torch.LongTensor(rules_gen).cuda()  # rules used for returning best rule
        self.rules_exp = tuple(rules_exp)  # rules without padding

    @property
    def num_rule(self):
        return self.rules.size(1)

    # Finding pseudo-groundings for a specific (h, r)
    def pgnd(self, h, i, num=None, rgnd=None):
        if num is None:
            num = self.arg('pgnd_num')

        key = (h, self.r, tuple(self.rules_exp[i]))
        if key in self.pgnd_buffer:
            return self.pgnd_buffer[key]

        with torch.no_grad():
            ans_can = self.arg('answer_candidates', apply=lambda x: x)
            if ans_can is not None:
                ans_can = ans_can.cuda()

            rule_embed = self.rotate.embed(h, self.tmp__rule_embed[i])
            if ans_can is None:
                dist = self.rotate.dist(rule_embed, self.rotate.entity_embed)
            else:
                can_dist = self.rotate.dist(rule_embed, self.rotate.entity_embed[ans_can])
                # dist = torch.zeros(self.E).cuda() + 1e10
                dist = torch.cuda.FloatTensor(self.E) * 0. + 1e10
                dist[ans_can] = can_dist

            if rgnd is not None:
                # print(len(rgnd), dist.size())
                dist[torch.LongTensor(rgnd).cuda()] = 1e10
            ret = torch.arange(self.E).cuda()[dist <= self.rotate.gamma]

            dist[ret] = 1e10
            num = min(num, dist.size(0) - len(rgnd)) - ret.size(-1)
            if num > 0:
                tmp = dist.topk(num, dim=0, largest=False, sorted=False)[1]
                ret = torch.cat([ret, tmp], dim=0)

        self.pgnd_buffer[key] = ret
        ##########
        # print(h, sorted(ret.cpu().numpy().tolist()))
        return ret

    # Calculate score in formula 17. A sparse matrix is given with column_idx=crule, row_idx=centity. Returns score in (17) in paper, as the value of the sparse matrix.
    def cscore(self, rule_embed, crule, centity, cweight, use_rotate=False):
        if not use_rotate:
            return torch.ones(len(crule)).cuda() * cweight
        else:
            score = self.rotate.compare(rule_embed, self.rotate.entity_embed, crule, centity)
            score = (self.rotate.gamma - score).sigmoid()
            if self.arg('drop_neg_gnd'):
                score = score * (score >= 0.5)
            score = score * cweight
            return score

    # Returns the rule's value in (16)
    def rule_value(self, batch, weighted=False):
        num_rule = self.num_rule
        h, t_list, mask, crule, centity, cweight = self.load_batch(batch)
        with torch.no_grad():

            rule_embed = self.rotate.embed(h, self.tmp__rule_embed)
            cscore = self.cscore(rule_embed, crule, centity, cweight, use_rotate=self.arg('filter_with_rotate'))

            indices = torch.stack([crule, centity], 0)

            def cvalue(cscore):
                if cscore.size(0) == 0:
                    # return torch.zeros(num_rule).cuda()
                    return torch.cuda.FloatTensor(num_rule) * 0.
                return torch.sparse.sum(torch.sparse.FloatTensor(
                    indices,
                    cscore,
                    torch.Size([num_rule, self.E])
                ).cuda(), -1).to_dense()


            pos = cvalue(cscore * mask[centity])
            neg = cvalue(cscore * ~mask[centity])
            score = cvalue(cscore)
            num = cvalue(cweight).clamp(min=0.001)

            pos_num = cvalue(cweight * mask[centity]).clamp(min=0.001)
            neg_num = cvalue(cweight * ~mask[centity]).clamp(min=0.001)


            def eval_ctx(local_ctx):
                local_ctx['self'] = self
                return lambda x: eval(x, globals(), local_ctx)

            value = self.arg('rule_value_def', apply=eval_ctx(locals()))

            if weighted:
                value *= len(t_list)

            if hasattr(self, 'tmp__rule_value'):
                self.tmp__rule_value += value
                self.tmp__num_init += len(t_list)

        return value

    # Choose rules, which has top `num_samples` of `value` and has a non negative `nonneg`
    def choose_rules(self, value, nonneg=None, num_samples=None, return_mask=False):
        if num_samples is None:
            num_samples = self.arg('max_best_rules')
        ################
        # print(f"choose_rules num = {num_samples}")
        with torch.no_grad():
            num_rule = value.size(-1)
            topk = value.topk(min(num_samples - 1, num_rule), dim=0, largest=True, sorted=False)[1]
            cho = torch.zeros(num_rule).bool().cuda()
            cho[topk] = True
            if nonneg is not None:
                cho[nonneg < 0] = False

        if return_mask:
            return cho
        return mask2list(cho)

    # For a new relation, init rule weights and choose rules
    def relation_init(self, r: int = None, rule_file: str = None, rules: List[Tuple[Tuple[int], float, int]] = None, prior=None, force_init_weight=False):
        print = self.print
        if r is not None:
            self.r = r
        r = self.r
        if rules is None:
            assert rule_file is not None
            rules: List[Tuple[Tuple[int], float, int]] = [((r,), 1., -1)]
            rule_set = set([tuple(), (r,)])
            with open(rule_file) as file:
                for i, line in enumerate(file):
                    try:
                        path, prec = line.split('\t')
                        path = tuple(map(int, path.split()))
                        prec = float(prec.split()[0])

                        if not (prec >= 0.0001):
                            # to avoid negative and nan
                            prec = 0.0001

                        if path in rule_set:
                            continue
                        rule_set.add(path)
                        if len(path) <= self.arg('max_rule_len'):
                            rules.append((path, prec, i))
                    except:
                        continue
                
            if self.arg("filter_with_recall"):
                rules = sorted(rules, key=lambda x: (x[1], x[2]), reverse=True)[:self.arg('max_beam_rules')]
            else:
                rules = sorted(rules, key=lambda x: (x[1], x[2]), reverse=True)
            print(f"Loaded from file: |rules| = {len(rules)} max_rule_len = {self.arg('max_rule_len')}")
            x = torch.tensor([prec for _, prec, _ in rules]).cuda()
            prior = -torch.log((1 - x.float()).clamp(min=1e-6))
            # prior = x
            rules: List[Tuple[int]] = [path for path, _, _ in rules]
        else:
            assert prior is not None
        
        mov = int((self.R - 1) / 2)
        
        if self.arg("enumerate_symmetric_rule"):
            rules = []
            for i in range(self.R - 1):
                if i != self.r and (i - mov) != self.r:
                    if i >= mov:
                        rules.append((self.r, i, i - mov))
                        rules.append((i, i - mov, self.r))
                    else:
                        rules.append((self.r, i, i + mov))
                        rules.append((i, i + mov, self.r))
            prior = torch.ones(len(rules)).cuda()

        self.prior = prior

        
        if self.arg("only_symmetric_rule"):
            new_rules = []
            new_prior = []
            for i, rule in enumerate(rules):
                if (len(rule) == 3 and rule[0] == r and abs(rule[1] - rule[2]) == mov) or (len(rule) == 3 and rule[2] == r and abs(rule[0] - rule[1]) == mov):
                    new_rules.append(rule)
                    new_prior.append(prior[i].item())
            
            rules = new_rules
            self.prior = torch.tensor(new_prior).cuda()
        
        if self.arg("without_symmetric_rule"):
            new_rules = []
            new_prior = []
            for i, rule in enumerate(rules):
                if not (len(rule) == 3 and rule[0] == r and abs(rule[1] - rule[2]) == mov) and not (len(rule) == 3 and rule[2] == r and abs(rule[0] - rule[1]) == mov):
                    new_rules.append(rule)
                    new_prior.append(prior[i].item())
            rules = new_rules
            self.prior = torch.tensor(new_prior).cuda()
            # self.prior = self.prior[torch.tensor([i for i in range(len(self.prior)) if (len(rules[i]) == 3 and rules[i][0] == r and (rules[i][1] == (rules[i][2] - mov) or (rules[i][1] - mov) == rules[i][2])) or len(rules[i]) == 1]) or (len(rules[i]) == 3 and rules[i][2] == r and (rules[i][0] == (rules[i][1] - mov) or (rules[i][0] - mov) == rules[i][1]))]
            # rules = [rule for rule in rules if (len(rule) == 3 and rule[0] == r and (rule[1] == (rule[2] - mov) or (rule[1] - mov) == rule[2])) or len(rule) == 1 or (len(rule) == 3 and rule[2] == r and (rule[0] == (rule[1] - mov) or (rule[0] - mov) == rule[1]))]
        self.set_rules(rules)

        num_rule = self.num_rule
        with torch.no_grad():
            # self.tmp__rule_value = torch.zeros(num_rule).cuda()
            self.tmp__rule_value = torch.cuda.FloatTensor(num_rule) * 0.
            self.tmp__rule_embed = self.rule_embed(force=True).detach()  # embedding of each rule
            self.tmp__num_init = 0
            if not self.arg('param_relation_embed'):
                self._rule_embed = self.tmp__rule_embed

        init_weight = force_init_weight or not self.arg('init_weight_with_prior')

        if init_weight:
            for batch in self.make_batchs(init=True):
                self.rule_value(batch, weighted=True)

        # choose args.max_rules of rules to train the model
        with torch.no_grad():
            if torch.isnan(self.tmp__rule_value).sum():
                self.tmp__rule_value[torch.isnan(self.tmp__rule_value)] = 0
            value = self.tmp__rule_value / max(self.tmp__num_init, 1) + self.arg('prior_coef') * self.prior
            nonneg = self.tmp__rule_value
            if self.arg('use_neg_rules') or not init_weight:
                nonneg = None
            # construct negative examples
            if self.arg('filter_rule'):
                cho = self.choose_rules(value, num_samples=self.arg('max_rules'), nonneg=nonneg, return_mask=True)

                cho[0] = True
                cho_list = mask2list(cho).detach().cpu().numpy().tolist()
                value_list = value.detach().cpu().numpy().tolist()
                cho_list = sorted(cho_list,
                                key=lambda x: (x == 0, value_list[x]), reverse=True)
                assert cho_list[0] == 0
                cho = torch.LongTensor(cho_list).cuda()

                value = value[cho]
                self.tmp__rule_value = self.tmp__rule_value[cho]
                self.prior = self.prior[cho]
                self.rules = self.rules[:, cho]
                self.rules_gen = self.rules_gen[cho]
                self.rules_exp = [self.rules_exp[x] for x in cho_list]

        if init_weight:
            weight = self.tmp__rule_value
        else:
            weight = self.prior

        print(f"weight_init: num = {self.num_rule} [{weight.min().item()}, {weight.max().item()}]")
        weight = weight.clamp(min=0.0001)
        weight /= weight.max()
        weight[0] = 1.0
        self.rule_weight_raw = torch.nn.Parameter(weight)

        del self.tmp__rule_value
        del self.tmp__num_init

        with torch.no_grad():
            self.tmp__rule_embed = self.rule_embed(force=True).detach()
            if not self.arg('param_relation_embed'):
                self._rule_embed = self.tmp__rule_embed

        self.make_batchs()

        del self.tmp__rule_embed

    # Default arguments for predictor
    def set_args(self, args):
        self._args = dict()
        def make_str(v):
            if isinstance(v, int):
                return True
            if isinstance(v, float):
                return True
            if isinstance(v, bool):
                return True
            if isinstance(v, str):
                return True
            return False

        for k, v in args.items():
            self._args[k] = str(v) if make_str(v) else v

    def arg(self, name, apply=None):
        # print(self._args[name])
        v = self._args[name]
        if apply is None:
            if v is None:
                return None
            return eval(v)
        return apply(v)

    @property
    def rule_weight(self):
        return self.rule_weight_raw

    def forward(self, batch):
        # 4ms in total
        E = self.E
        R = self.R

        rule_weight = self.rule_weight
        # 0.9ms
        _rule_embed = self.rule_embed()
        rule_embed = []
        crule = []
        crule_weight = []
        centity = []
        cweight = []
        csplit = [0]
        
        for single in batch:
            _h, _, _, _crule, _centity, _cweight = self.load_batch(single)
            if _crule.size(0) == 0:
                csplit.append(csplit[-1])
                continue
            crule.append(_crule + len(rule_embed) * self.num_rule)
            crule_weight.append(rule_weight.index_select(0, _crule))
            centity.append(_centity)
            cweight.append(_cweight)
            # 0.7ms
            rule_embed.append(self.rotate.embed(_h, _rule_embed))
            csplit.append(csplit[-1] + _crule.size(-1))

        if len(crule) == 0:
            crule = torch.tensor([]).long().cuda()
            crule_weight = torch.tensor([]).cuda()
            centity = torch.tensor([]).long().cuda()
            cweight = torch.tensor([]).cuda()
            rule_embed = torch.tensor([]).cuda()
            cscore = torch.tensor([]).cuda()
        else:
            crule = torch.cat(crule, dim=0)
            crule_weight = torch.cat(crule_weight, dim=0)
            centity = torch.cat(centity, dim=0)
            cweight = torch.cat(cweight, dim=0)
            rule_embed = torch.cat(rule_embed, dim=0)
            # 0.3ms
            
            cscore = self.cscore(rule_embed, crule, centity, cweight, use_rotate=self.arg("train_with_rotate")) * (crule_weight if not self.arg("without_rule_weight") else 1)




        loss = torch.tensor(0.0).cuda().requires_grad_() + 0.0
        result = []

        for i, single in enumerate(batch):
            _h, t_list, mask, _crule, _centity, _cweight = self.load_batch(single)
            if _crule.size(0) != 0:
                # 0.5 - 1ms
                score = torch.cuda.FloatTensor(E, self.num_rule) * 0.
                crange = torch.arange(csplit[i], csplit[i + 1]).cuda()
                
                score[_centity, _crule] = self.index_select(cscore, crange)
                score = score.sum(1)
            else:
                # score = torch.zeros(self.E).cuda()
                score = torch.cuda.FloatTensor(self.E) * 0.
                score.requires_grad_()

            ans_can = self.arg('answer_candidates', apply=lambda x: x)
            if ans_can is not None:
                ans_can = ans_can.cuda()
                score = self.index_select(score, ans_can)
                mask = self.index_select(mask, ans_can)

                # map_arr = -torch.ones(self.E).long().cuda()
                map_arr = - (torch.cuda.LongTensor(self.E) * 0 + 1)
                map_arr[ans_can] = torch.arange(ans_can.size(0)).long().cuda()
                map_arr = map_arr.detach().cpu().numpy().tolist()
                map_fn = lambda x: map_arr[x]
                t_list = list(map(map_fn, t_list))

            if self.recording:
                self.record.append((score.cpu(), mask, t_list))

            elif not self.training:
                for t in t_list:
                    result.append(self.metrics.apply(score, mask.bool(), t))

            if score.dim() == 0:
                continue

            score = score.softmax(dim=-1)
            neg = score.masked_select(~mask.bool())

            loss += neg.sum()

            for t in t_list:
                s = score[t]
                wrong = (neg > s)
                loss += ((neg - s) * wrong).sum() / wrong.sum().clamp(min=1)

        return loss / len(batch), self.metrics.summary(result)

    def _evaluate(self, valid_batch, batch_size=None):
        model = self
        if batch_size is None:
            batch_size = self.arg('predictor_batch_size') * self.arg('predictor_eval_rate')
        print_epoch = self.arg('predictor_print_epoch') * self.arg('predictor_eval_rate')
        # print(print_epoch)

        self.eval()
        with torch.no_grad():
            result = Metrics.zero_value()
            for i in range(0, len(valid_batch), batch_size):
                cur = model(valid_batch[i: i + batch_size])[1]
                result = Metrics.merge(result, cur)
                if i % print_epoch == 0 and i > 0:
                    print(f"eval #{i}/{len(valid_batch)}")
        return result

    # make a single batch, find groundings and pseudo-groundings
    def _make_batch(self, h, t_list, answer=None, rgnd_buffer=None, use_pgnd=False):
        # print("make_batch in")
        flag = False
        if answer is None:
            flag = True
            answer = t_list
        if rgnd_buffer is None:
            rgnd_buffer = self.rgnd_buffer
        crule = []
        centity = []
        cweight = []
        gnd = []
        max_pgnd_rules = self.arg('max_pgnd_rules')
        if max_pgnd_rules is None:
            max_pgnd_rules = self.arg('max_rules')

        for i, rule in enumerate(self.rules_exp):
            # print(f"iter i = {i} / {len(self.rules_exp)}")
            if (i != 0 or (i == 0 and self.rules_exp[i] != (self.r,))) and not self.arg('disable_gnd'):
                key = (h, self.r, rule)
                if key in rgnd_buffer:
                    rgnd = rgnd_buffer[key]
                else:
                    # print("gnd in")
                    rgnd = groundings.groundings(h, rule)

                    ans_can = self.arg('answer_candidates', apply=lambda x: x)
                    if ans_can is not None:
                        ans_can = set(ans_can.cpu().numpy().tolist())
                        rgnd = list(filter(lambda x: x in ans_can, rgnd))
                    rgnd_buffer[key] = rgnd

                ones = torch.ones(len(rgnd))
                centity.append(torch.LongTensor(rgnd))
                crule.append(ones.long() * i)
                cweight.append(ones)
            else:
                rgnd = []

            gnd.append(rgnd)

            if use_pgnd:
                if (i == 0 and self.rules_exp[i] == (self.r,)) and self.arg('disable_selflink'):
                    continue
                if i >= max_pgnd_rules:
                    continue
                num = self.arg('pgnd_num') * self.arg('pgnd_selflink_rate' if (i == 0 and self.rules_exp[i] == (self.r,)) else 'pgnd_nonselflink_rate')
                # num = 0
                pgnd = self.pgnd(h, i, num, gnd[i])
                
                ones = torch.ones(len(pgnd))
                centity.append(pgnd.long().cpu())
                crule.append(ones.long() * i)
                cweight.append(ones * self.arg('pgnd_weight'))

        # print("iter done")
        if len(crule) == 0:
            crule = torch.tensor([]).long().cuda()
            centity = torch.tensor([]).long().cuda()
            cweight = torch.tensor([]).float().cuda()
        else:
            crule = torch.cat(crule, dim=0)
            centity = torch.cat(centity, dim=0)
            cweight = torch.cat(cweight, dim=0)

        #################
        # print("work", answer)

        # print("make_batch out")

        return h, t_list, list2mask(answer, self.E), crule, centity, cweight

    # make all batchs
    def make_batchs(self, init=False):
        print = self.print
        # if r is not None:
        # 	self.r = r
        dataset = self.dataset
        graph = build_graph(dataset['train'], self.E, self.R)
        graph_test = build_graph(dataset['train'] + dataset['valid'], self.E, self.R)

        def filter(tri):
            a = defaultdict(lambda: [])
            for h, r, t in tri:
                if r == self.r:
                    a[h].append(t)
            return a

        train = filter(dataset['train'])
        valid = filter(dataset['valid'])
        test = filter(dataset['test'])

        answer_valid = defaultdict(lambda: [])
        answer_test = defaultdict(lambda: [])
        for a in [train, valid]:
            for k, v in a.items():
                answer_valid[k] += v
                answer_test[k] += v
        for k, v in test.items():
            answer_test[k] += v

        if len(train) > self.arg('max_h'):
            from random import shuffle
            train = list(train.items())
            shuffle(train)
            train = train[:self.arg('max_h')]
            train = {k: v for (k, v) in train}

        print_epoch = self.arg('predictor_init_print_epoch')

        self.train_batch = []
        self.valid_batch = []
        self.test_batch = []

        groundings.use_graph(graph)

        if init:
            def gen_init(self, train, print_epoch):
                for i, (h, t_list) in enumerate(train.items()):
                    if i % print_epoch == 0:
                        print(f"init_batch: {i}/{len(train)}")
                    yield self._make_batch(h, t_list, use_pgnd=self.arg("filter_with_pgnd"))

            return gen_init(self, train, print_epoch)

        for i, (h, t_list) in enumerate(train.items()):
            if i % print_epoch == 0:
                print(f"train_batch: {i}/{len(train)}")
            batch = list(self._make_batch(h, t_list, use_pgnd=self.arg("train_with_pgnd")))
            for t in t_list:
                batch[1] = [t]
                self.train_batch.append(tuple(batch))

        for i, (h, t_list) in enumerate(valid.items()):
            if i % print_epoch == 0:
                print(f"valid_batch: {i}/{len(valid)}")
            self.valid_batch.append(self._make_batch(h, t_list, answer=answer_valid[h], use_pgnd=self.arg("train_with_pgnd")))

        groundings.use_graph(graph_test)
        for i, (h, t_list) in enumerate(test.items()):
            if i % print_epoch == 0:
                print(f"test_batch: {i}/{len(test)}")
            self.test_batch.append(
                self._make_batch(h, t_list, answer=answer_test[h], rgnd_buffer=self.rgnd_buffer_test, use_pgnd=self.arg("train_with_pgnd")))

    def train_model(self):
        # self.make_batchs()
        train_batch = self.train_batch
        valid_batch = self.valid_batch
        test_batch = self.test_batch
        model = self
        print = self.print
        batch_size = self.arg('predictor_batch_size')
        num_epoch = self.arg('predictor_num_epoch')  # / batch_size
        lr = self.arg('predictor_lr')  # * batch_size
        print_epoch = self.arg('predictor_print_epoch')
        valid_epoch = self.arg('predictor_valid_epoch')

        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.0)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epoch, eta_min=lr / 5)

        for name, param in self.named_parameters():
            print(f"Model Parameter: {name} ({param.type()}:{param.size()})")

        self.best = Metrics.init_value()
        self.best_model = self.state_dict()

        def train_step(batch):
            self.train()
            loss, _ = self(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            return loss

        def metrics_score(result):
            result = Metrics.pretty(result)
            mr = result['mr']
            mrr = result['mrr']
            h1 = result['h1']
            h3 = result['h3']
            h10 = result['h10']

            def eval_ctx(local_ctx):
                local_ctx['self'] = self
                return lambda x: eval(x, globals(), local_ctx)

            return self.arg('metrics_score_def', apply=eval_ctx(locals()))

        def format(result):
            s = ""
            for k, v in Metrics.pretty(result).items():
                if k == 'num':
                    continue
                s += k + ":"
                s += "%.4lf " % v
            return s

        def valid():
            result = self._evaluate(valid_batch)
            updated = False
            if metrics_score(result) > metrics_score(self.best):
                updated = True
                self.best = result
                self.best_model = copy.deepcopy(self.state_dict())
            print(f"valid = {format(result)} {'updated' if updated else ''}")
            return updated, result

        last_update = 0
        cum_loss = 0
        valid()

        # relation_embed_init = self.rotate.relation_embed.clone()

        if len(train_batch) == 0:
            num_epoch = 0
        for epoch in range(1, num_epoch + 1):
            if epoch % max(1, len(train_batch) // batch_size) == 0:
                from random import shuffle
                shuffle(train_batch)
            batch = [train_batch[(epoch * batch_size + i) % len(train_batch)] for i in range(batch_size)]
            loss = train_step(batch)
            cum_loss += loss.item()

            if epoch % print_epoch == 0:
                lr_str = "%.2e" % (opt.param_groups[0]['lr'])
                print(f"train_predictor #{epoch} lr = {lr_str} loss = {cum_loss / print_epoch}")
                cum_loss *= 0

            if epoch % valid_epoch == 0:
                if valid()[0]:
                    last_update = epoch
                elif epoch - last_update >= self.arg('predictor_early_break_rate') * num_epoch:
                    print(f"Early break: Never updated since {last_update}")
                    break
                if 1 - 1e-6 < Metrics.pretty(self.best)['mr'] < 1 + 1e-6:
                    print(f"Early break: Perfect")
                    break

        with torch.no_grad():
            self.load_state_dict(self.best_model)
            # self.rotate.relation_embed *= 0
            # self.rotate.relation_embed += relation_embed_init
            # self.rule_weight_raw[0] += 1000.0
            valid()

        self.load_state_dict(self.best_model)
        best = self.best
        if self.arg('record_test'):
            backup = self.recording
            self.record = []
            self.recording = True
        test = self._evaluate(test_batch)
        # self.print_score()

        if self.arg('record_test'):
            self.recording = backup

        print("__Best_Valid__\t" + ("\t".join([str(self.r), str(int(best[0]))] + list(map(lambda x: "%.4lf" % x, best[1:])))))
        print("__Test__\t" + ("\t".join([str(self.r), str(int(test[0]))] + list(map(lambda x: "%.4lf" % x, test[1:])))))

        return best, test

    def tmp(self):
        outputs = dict()
        for i, rule in enumerate(self.rules_exp):
            outputs[str(rule)] = {"weight": self.rule_weight[i].item(), "head": {}}

        for example in self.rotate_weight_buffer.keys():
            head, rule, tail = eval(example)
            if head not in outputs[str(tuple(rule))]["head"]:
                outputs[str(tuple(rule))]["head"][head] = {"labels": {}, "others": {}}
            else:
                if [head, self.r, tail] in self.dataset['test']:
                    outputs[str(tuple(rule))]["head"][head]["labels"][tail] = self.rotate_weight_buffer[example].item()
                else:
                    outputs[str(tuple(rule))]["head"][head]["others"][tail] = self.rotate_weight_buffer[example].item()

    def print_score(self):
        # outputs = dict()
        # rules_exp = [list(tmp) for tmp in self.rules_exp]
        data2label = {}
        for split in ["train", "valid", "test"]:
            for example in self.dataset[split]:
                data2label[str(example)] = split

        with open(self.args.output_dir + f"/relation-{self.r}-rotate-score.txt", "w") as file:
            file.write(f"rule\trule weight\thead\ttail\tlabel\tpath weight\n")
            _, indices = self.rule_weight.sort(descending=True)
            for index in indices:
                rule = self.rules_exp[index.item()]
                if str(list(rule)) in self.rotate_weight_buffer:
                    for a, b in sorted(self.rotate_weight_buffer[str(list(rule))].items(), key=lambda x: x[1], reverse=True):
                        head, tail = eval(a)
                        value = b
                        if str([head, self.r, tail]) in data2label:
                            label = data2label[str([head, self.r, tail])]
                        else:
                            label = "false"

                        head = self.scorer.entity2text[head]
                        tail = self.scorer.entity2text[tail]
                        file.write(f"{str([self.scorer.relation2text[r] for r in rule])}\t{round(self.rule_weight[index].item(), 3)}\t{head}\t{tail}\t{label}\t{round(value.item(), 3)}\n")
                    else:
                        file.write(f"{str([self.scorer.relation2text[r] for r in rule])}\t{round(self.rule_weight[index].item(), 3)}\tNo Example\n")

    def find_symmetric_rule(self, h, t_list):
        rules = [[]] * len(t_list)

        if len(t_list) == 0:
            return rules

        mov = int((self.R - 1) / 2)
        for i in range(mov):
            t1 = [_[2] for _ in self.dataset["train"] if _[0] == h and _[1] == self.r]

            t2 = [_[2] for _ in self.dataset["train"] if _[1] == i and _[0] in t1]
            t3 = [_[2] for _ in self.dataset["train"] if _[1] == i + mov and _[0] in t2]
            
            for j, t in enumerate(t_list):
                if t in t3:
                    rules[j].append((self.r, i, i + mov))

            t2 = [_[2] for _ in self.dataset["train"] if _[1] == i + mov and _[0] in t1]
            t3 = [_[2] for _ in self.dataset["train"] if _[1] == i and _[0] in t2]
            
            for j, t in enumerate(t_list):
                if t in t3:
                    rules[j].append((self.r, i + mov, i))
        
            t1 = [_[2] for _ in self.dataset["train"] if _[0] == h and _[1] == i]
            t2 = [_[2] for _ in self.dataset["train"] if _[1] == i + mov and _[0] in t1]
            t3 = [_[2] for _ in self.dataset["train"] if _[1] == self.r and _[0] in t2]

            for j, t in enumerate(t_list):
                if t in t3:
                    rules[j].append((i, i + mov, self.r))

            t1 = [_[2] for _ in self.dataset["train"] if _[1] == i + mov and _[0] == h]
            t2 = [_[2] for _ in self.dataset["train"] if _[1] == i and _[0] in t1]
            t3 = [_[2] for _ in self.dataset["train"] if _[1] == self.r and _[0] in t2]

            for j, t in enumerate(t_list):
                if t in t3:
                    rules[j].append((i + mov, i, self.r))
            
        return rules

