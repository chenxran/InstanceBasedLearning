from datetime import datetime
import sys
import os
import argparse
from merge_results import calc_result
import rule_sample
from knowledge_graph_utils import *
from datetime import datetime
# from model_rnnlogic import RNNLogic

import multiprocessing as mp
from model import RNNLogic
import numpy as np
import random

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

    # basic arguments
    parser.add_argument("--task", type=str, default=None, required=True)
    parser.add_argument("--write_log_to_console", type=bool, default=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--hop", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--parallel", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=12)


    # model & generator arguments
    parser.add_argument("--num_em_epoch", type=int, default=3)
    parser.add_argument("--sample_print_epoch", type=int, default=100)
    parser.add_argument("--max_beam_rules", type=int, default=3000)
    parser.add_argument("--generator_embed_dim", type=int, default=512)
    parser.add_argument("--generator_hidden_dim", type=int, default=256)
    parser.add_argument("--generator_lr", type=float, default=1e-3)
    parser.add_argument("--generator_num_epoch", type=int, default=10000)
    parser.add_argument("--generator_print_epoch", type=int, default=100)
    parser.add_argument("--init_weight_boot", type=bool, default=False)
    parser.add_argument("--RotatE", type=str, default=None)
    parser.add_argument("--rotate_pretrained", type=str, default=None)
    parser.add_argument("--without_rule_weight", type=bool, default=False)
    parser.add_argument("--filter_rule", type=bool, default=False)
    parser.add_argument("--filter_with_rotate", type=bool, default=False)
    parser.add_argument("--train_with_rotate", type=bool, default=False)
    parser.add_argument("--only_symmetric_rule", type=bool, default=False)
    parser.add_argument("--enumerate_symmetric_rule", type=bool, default=False)
    parser.add_argument("--without_symmetric_rule", type=bool, default=False)
    parser.add_argument("--filter_with_pgnd", type=bool, default=False)
    parser.add_argument("--train_with_pgnd", type=bool, default=False)

    # predictor arguments
    parser.add_argument("--max_rules", type=int, default=1000)
    parser.add_argument("--max_rule_len", type=int, default=None)
    parser.add_argument("--max_h", type=int, default=5000)
    parser.add_argument("--max_best_rules", type=int, default=300)
    parser.add_argument("--param_relation_embed", type=bool, default=False)
    parser.add_argument("--param_entity_embed", type=bool, default=False)
    parser.add_argument("--init_weight_with_prior", type=bool, default=False)
    parser.add_argument("--prior_coef", type=float, default=0.01)
    parser.add_argument("--use_neg_rules", type=bool, default=False)
    parser.add_argument("--disable_gnd", type=bool, default=False)
    parser.add_argument("--disable_selflink", type=bool, default=False)
    parser.add_argument("--drop_neg_gnd", type=bool, default=False)
    parser.add_argument("--pgnd_num", type=int, default=256)
    parser.add_argument("--pgnd_selflink_rate", type=int, default=8)
    parser.add_argument("--pgnd_nonselflink_rate", type=int, default=0)
    parser.add_argument("--pgnd_weight", type=float, default=0.1)

    parser.add_argument("--max_pgnd_rules", type=int, default=None)
    parser.add_argument("--predictor_num_epoch", type=int, default=200000)
    parser.add_argument("--predictor_early_break_rate", type=float, default=0.2)  # originally 1 / 5
    parser.add_argument("--predictor_lr", type=float, default=5e-5)
    parser.add_argument("--predictor_batch_size", type=int, default=1)
    parser.add_argument("--predictor_print_epoch", type=int, default=100)
    parser.add_argument("--predictor_init_print_epoch", type=int, default=10)
    parser.add_argument("--predictor_valid_epoch", type=int, default=100)
    parser.add_argument("--predictor_eval_rate", type=int, default=4)
    parser.add_argument("--rule_value_def", type=str, default='(pos - neg) / num') 
    parser.add_argument("--metrics_score_def", type=str, default='(mrr+0.9*h1+0.8*h3+0.7*h10+0.01/max(1,mr), mrr, mr, h1, h3, h10, -mr)')  
    parser.add_argument("--answer_candidates", type=bool, default=None)
    parser.add_argument("--record_test", type=bool, default=False)

    parser.add_argument("--filter_with_recall", type=bool, default=False)
    args = parser.parse_args()

    # prepare data dir and output dir
    args.data_dir = f"RNNLogic/data/{args.task}"
    args.output_dir = f"RNNLogic/logs/{args.task}"

    if args.enumerate_symmetric_rule or args.only_symmetric_rule:
        args.max_rule_len = 3

    # default hyper-parameters in RNNLogic
    if args.RotatE is None:
        if args.task == "FB15k-237":
            args.RotatE = "RotatE_500"
            if args.max_rule_len is None:
                args.max_rule_len = 4
        elif args.task == "FB15k-237_0.1":
            args.RotatE = "RotatE_500"
            if args.max_rule_len is None:
                args.max_rule_len = 4           
        elif args.task == "wn18rr":
            args.RotatE = "RotatE_200"
            if args.max_rule_len is None:
                args.max_rule_len = 5
        elif args.task == "wn18rr_0.6":
            if args.max_rule_len is None:
                args.RotatE = "RotatE_200"
            if args.max_rule_len is None:
                args.max_rule_len = 5
        elif args.task == "kinship":
            args.RotatE = "RotatE_2000"
            if args.max_rule_len is None:
                args.max_rule_len = 3
        elif args.task == "umls":
            args.RotatE = "RotatE_1000"
            if args.max_rule_len is None:
                args.max_rule_len = 3
        elif args.task == "umls_0.03":
            args.RotatE = "RotatE_1000"
            if args.max_rule_len is None:
                args.max_rule_len = 3
        else:
            raise NotImplementedError

    # sanity check
    if args.rotate_pretrained is None:
        args.rotate_pretrained = f"{args.data_dir}/{args.RotatE}"

    return args


def train(config, dataset, r):
    log_filename = f"{config.output_dir}/log_{r}.txt"

    def new_print(*args, **kwargs):
        print(*args, **kwargs)
        with open(log_filename, 'a') as file:
            file.write(' '.join([str(_) for _ in args]) + (kwargs["end"] if "end" in kwargs else '\n'))

    model = RNNLogic(dataset, config, print=new_print)
    
    valid, test = model.train_model(
        r,
        rule_file=f"{config.data_dir}/Rules/rules_{config.max_rule_len}_{r}.txt",
        model_file=f"{config.output_dir}/model_{r}.pth"
    )
    return valid, test


def main(args):
    with open(f"{args.output_dir}/config.txt", 'w') as file:
        for k, v in vars(args).items():
            file.write(f"{k}: {v}\n")

    # Step 1: Load dataset
    dataset = load_dataset(args.data_dir)
    set_random_seed(42)

    # Step 2: Generate rules
    # Note: This step only needs to do once.
    args.end = dataset['R'] if args.end is None else args.end
    itr = list(range(args.start, args.end, args.hop))
    
    rule_sample.use_graph(dataset_graph(dataset, 'train'))
    for r in itr:
        if not os.path.exists(f"{args.data_dir}/Rules/rules_{args.max_rule_len}_{r}.txt"):
        # Usage: rule_sample.sample(relation, dict: rule_len -> num_per_sample, num_samples, ...)
            if args.max_rule_len == 5:
                rules = rule_sample.sample(r, {1: 1, 2: 100, 3: 100, 4: 100, 5: 100}, 1000, num_threads=12, samples_per_print=100)
            elif args.max_rule_len == 4:
                rules = rule_sample.sample(r, {1: 1, 2: 100, 3: 100, 4: 100}, 1000, num_threads=12, samples_per_print=100)
            elif args.max_rule_len == 3:
                rules = rule_sample.sample(r, {1: 1, 2: 100, 3: 100}, 1000, num_threads=12, samples_per_print=100)
            elif args.max_rule_len == 1:
                rules = rule_sample.sample(r, {1: 10}, 10000, num_threads=12, samples_per_print=100)

            if not os.path.exists(f"{args.data_dir}/Rules"):
                os.makedirs(f"{args.data_dir}/Rules")
            rule_sample.save(rules, f"{args.data_dir}/Rules/rules_{args.max_rule_len}_{r}.txt")

    for r in itr:
        valid, test = train(args, dataset, r)


if __name__ == "__main__":
    config = parse_args()
    old_print = print
    time = datetime.now()
    os.makedirs(f"{config.output_dir}/{time}/")
    config.output_dir = f'{config.output_dir}/{time}'
    valid, test = main(config)
    
    def merge_files(path):
        count = 0
        files = os.listdir(path)
        for file in files:
            if file.endswith(".txt") and file.startswith("log_"):
                f = open(path + "/" + file).read()
                log = open(path + "/" + 'train_log.txt', 'a+')
                log.write(f)
                count += 1
        return count

    num_relation = merge_files(config.output_dir)

    path = config.output_dir + "/train_log.txt"
    total = 0

    all_mr = 0
    all_mrr = 0
    all_h1 = 0
    all_h3 = 0
    all_h10 = 0

    for i in range(num_relation):
        with open(path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if "r = {} EM = 0 | __Test__".format(i) in line:
                    count, mr, mrr, h1, h3, h10 = [eval(tmp) for tmp in line.strip().split("\t")[-6:]]
                    total += count
                    all_mr += mr * count
                    all_mrr += mrr * count
                    all_h1 += h1 * count
                    all_h3 += h3 * count
                    all_h10 += h10 * count


    with open(config.output_dir + "/result.txt", "w") as file:
        file.write("Total: {}\n".format(total))
        file.write("MR: {}\n".format(all_mr / total))
        file.write("MRR: {}\n".format(all_mrr / total))
        file.write("H@1: {}\n".format(all_h1 / total))
        file.write("H@3: {}\n".format(all_h3 / total))
        file.write("H@10: {}\n".format(all_h10 / total))
