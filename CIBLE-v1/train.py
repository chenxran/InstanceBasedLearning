import argparse
from datetime import datetime

import os
import torch

from model import CIBLEDistMult, IBLEDistMult, ACTFN, IBLETransE, CIBLETransE, CIBLETransR, IBLETransR  # , MFRotatEModel, MFV4Model
from pykeen.datasets import UMLS, get_dataset, Kinships, FB15k237, WN18RR
from pykeen.losses import CrossEntropyLoss, MarginRankingLoss
from pykeen.pipeline import pipeline
from pykeen.regularizers import NoRegularizer, LpRegularizer, PowerSumRegularizer
import numpy as np

OPTIM = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    'adagrad': torch.optim.Adagrad,
    'adadelta': torch.optim.Adadelta,
    'rmsprop': torch.optim.RMSprop,
    'adamax': torch.optim.Adamax,
    'adamw': torch.optim.AdamW,
}

REG = {
    None: NoRegularizer,
    'none': NoRegularizer,
    'lp': LpRegularizer,
    'powersum': PowerSumRegularizer,
}

SCHEDULER = {
    None: None,
    'none': None,
    'step': torch.optim.lr_scheduler.StepLR,
    'exponential': torch.optim.lr_scheduler.ExponentialLR,
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'cosine_warm_restarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'cyclic': torch.optim.lr_scheduler.CyclicLR,
    'lambda': torch.optim.lr_scheduler.LambdaLR,
    'multistep': torch.optim.lr_scheduler.MultiStepLR,
}


LOSS = {
    "crossentropy": CrossEntropyLoss,
    "margin": MarginRankingLoss,
}


MODEL = {
    'IBLEDistMult': IBLEDistMult,
    'IBLETransR': IBLETransR,
    "CIBLEDistMult": CIBLEDistMult,
    "IBLETransE": IBLETransE,
    "CIBLETransE": CIBLETransE,
    "CIBLETransR": CIBLETransR,
}

DATASET = {
    "UMLS": UMLS,
    "Kinships": Kinships,
    "FB15k-237": FB15k237,
    "WN18RR": WN18RR,
}


def main(args):
    dataset_kwargs = dict(create_inverse_triples=True)
    if args.dataset in ["UMLS", "Kinships"]:
        dataset_kwargs["version"] = args.dataset_version
    dataset = DATASET[args.dataset](**dataset_kwargs)
    # dataset = get_dataset(dataset=args.dataset, dataset_kwargs=dataset_kwargs)

    save_path = f'/data/chenxingran/CIBLE/CIBLE-v1/results/{args.dataset}/{args.model}-{datetime.now()}'
    os.makedirs(save_path)
    # write argument to file
    with open(os.path.join(save_path, 'config.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')

    if args.model in ["IBLEDistMult", "IBLETransR", "IBLETransE", "CIBLETransR", "CIBLEDistMult", "CIBLETransR"]:
        if args.model in ["IBLEDistMult", "CIBLETransE", "IBLETransE", "CIBLEDistMult"]:
            model_kwargs = dict(
                args=args,
                triples_factory=dataset.training,
                embedding_dim=args.embedding_dim,
                random_seed=args.random_seed,
                loss=LOSS[args.loss],
                regularizer=REG[args.regularizer](
                    p=args.reg_p, 
                    weight=args.reg_weight
                ) if args.regularizer in ['lp', 'powersum'] else None,
            )
        elif args.model in ["IBLETransR", "CIBLETransR"]:
            model_kwargs = dict(
                args=args,
                triples_factory=dataset.training,
                entity_embedding_dim=args.entity_embedding_dim,
                relation_embedding_dim=args.relation_embedding_dim,
                random_seed=args.random_seed,
                loss=LOSS[args.loss],
                regularizer=REG[args.regularizer](
                    p=args.reg_p, 
                    weight=args.reg_weight
                ) if args.regularizer in ['lp', 'powersum'] else None,
            )
        model = MODEL[args.model](**model_kwargs)

        # write model to file
        with open(os.path.join(save_path, 'model.txt'), 'w') as f:
            for n, p in model.named_parameters():
                f.write(f'{n}: {p.size()}\n')

        pipeline_result = pipeline(
            dataset=dataset,
            model=model,
            training_kwargs=dict(
                num_epochs=args.epochs, 
                batch_size=args.batch_size
            ),
            random_seed=args.random_seed,
            training_loop=args.training_loop,
            optimizer=OPTIM[args.optimizer],
            optimizer_kwargs=dict(lr=args.learning_rate),
            stopper='early',
            stopper_kwargs=dict(
                frequency=5, 
                patience=3, 
                metric="both.realistic.inverse_harmonic_mean_rank", 
                relative_delta=0.002, 
            ),
            result_tracker='csv',
            result_tracker_kwargs=dict(
                name=f'{save_path}/result',
            ),
            evaluator_kwargs=dict(
                batch_size=64,
            ),
            evaluation_kwargs=dict(
                batch_size=64,
            ),
        )
    else:
        pipeline_result = pipeline(
            dataset=dataset,
            model=args.model,
            epochs=args.epochs,
            random_seed=args.random_seed,
            stopper='early',
            stopper_kwargs=dict(
                frequency=5, 
                patience=3, 
                metric="both.realistic.inverse_harmonic_mean_rank", 
                relative_delta=0.002, 
            ),
            result_tracker='csv',
            result_tracker_kwargs=dict(
                name=f'{save_path}/result',
            ),
            evaluator_kwargs=dict(
                batch_size=64,
            ),
            evaluation_kwargs=dict(
                batch_size=64,
            ),
        )

    if args.save:
        pipeline_result.save_to_directory(f"{save_path}/pipeline_result")


if __name__ == "__main__":
    # # set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mf-transe", help="model name")

    parser.add_argument("--dataset", type=str, default="UMLS", help="dataset name")
    parser.add_argument("--dataset_version", type=str, default="default", help="version of dataset (only used in UMLS and Kinships).")
    
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--embedding_dim", type=int, default=64, help="embedding dimension")

    parser.add_argument("--entity_embedding_dim", type=int, default=None, help="embedding dimension")
    parser.add_argument("--relation_embedding_dim", type=int, default=None, help="embedding dimension")

    parser.add_argument("--loss", type=str, default="crossentropy", help="loss function")
    parser.add_argument("--training_loop", type=str, default="LCWA", help="type of training loop")
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimizer")
    parser.add_argument("--regularizer", type=str, default="powersum", help="regularizer")
    parser.add_argument("--reg_weight", type=float, default=2, help="weight of regularizer loss")
    parser.add_argument("--reg_p", type=float, default=3.0, help="norm of regularizer")
    parser.add_argument("--activation", type=str, default="tanh", help="activation function")
    parser.add_argument("--selfmask", action='store_true', help="self-masking")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument("--save", action='store_true', help="Whether to save experimental result.")

    # argument for calculating identity matrix
    parser.add_argument("--not_share_entity_embedding", action='store_true')
    parser.add_argument("--not_share_relation_embedding", action='store_true')
    parser.add_argument("--diag_w", action='store_true')
    parser.add_argument("--w", action='store_true')
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--no_identity_matrix", action='store_true')
    parser.add_argument("--mlp", action='store_true')
    parser.add_argument("--intermediate_dim", type=int, default=None)

    # argument for mftranse
    parser.add_argument("--transe_weight", type=str, default=10, help="weight of transe loss")
    parser.add_argument("--bias", type=float, default=0.0, help="bias of identity matrix")
    parser.add_argument("--scoring_fct_norm", type=float, default=2.0, help="norm of transe loss")
    # parser.add_argument("--temperature", type=float, default=1.0, help="temperature")

    args = parser.parse_args()

    if args.transe_weight != "trainable":
        args.transe_weight = float(args.transe_weight)
    
    if args.model in ["CIBLETransR", "IBLETransR"]:
        args.normalize = True
    
    if args.entity_embedding_dim is None:
        args.entity_embedding_dim = args.embedding_dim
    if args.relation_embedding_dim is None:
        args.relation_embedding_dim = args.embedding_dim
    
    assert args.dataset_version in ["default", "rnnlogic", "neural-lp"]

    main(args)
