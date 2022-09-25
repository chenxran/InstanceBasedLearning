#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel  # , KGEV2Model

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

import math
import wandb
from datetime import datetime
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--freeze_entity_embedding', action='store_true', help='pretrained weight of model')
    parser.add_argument('--freeze_relation_embedding', action='store_true', help='pretrained weight of model')
    parser.add_argument('--pretrained', default=None, help='whether to used pretrained weight of model')
    parser.add_argument('--pretrained_path', type=str, default=None, help='pretrained weight of model')

    # hyper-parameter for CIBLE-RotatE
    parser.add_argument('--weight', type=float, default=1.0, help='weight')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--activation', type=str, default='none', help='activation function')
    parser.add_argument('--normalize', action='store_true', help='normalize')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature of score')
    parser.add_argument('--mlp', type=str, default="False", help='whether to use mlp')
    parser.add_argument('--intermediate_dim', type=int, default=None, help='temperature of score')    
    parser.add_argument('--r_type', type=str, default='diag')
    parser.add_argument('--im_cal', type=str, default='rotate')
    parser.add_argument('--method', type=str, default="default", help='negative sampling or global crossentropy')
    parser.add_argument('--pooling', type=str, default='sum')
    parser.add_argument('--sigmoid_rotate', type=str, default="False", help='sigmoid rotate')

    parser.add_argument('--training_epochs', type=int, default=None)
    parser.add_argument('--evaluate_strategy', type=str, default='steps')

    args = parser.parse_args(args)

    args.sigmoid_rotate = True if args.sigmoid_rotate.lower() == "true" else False
    args.mlp = True if args.mlp.lower() == "true" else False

    return args

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    # args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    
def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    wandb.init(project='cible', entity='chenxran', config=args)
    wandb.run.log_code('.')

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    wandb.log({"Step": step, **{f"{mode}/{k}": v for k, v in metrics.items()}})
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

        
def main(args):
    set_seed()
    # args.__dict__.update(json.load(open(os.path.join(args.pretrained_path, 'config.json'))))
    # args.save_path = "/data/chenxingran/CIBLE/CIBLE-v2/models"
    # args.data_path = "/data/chenxingran/CIBLE/CIBLE-v2/data/kinship"
    # args.im_cal = 'rotate'
    # args.sigmoid_rotate = True

    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    args.save_path = f"models/{args.model}-{args.data_path.split('/')[1]}-{datetime.now()}"
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    for arg in vars(args):
        print(arg, getattr(args, arg))

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    
    kge_model = KGEModel(
        args=args,
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        weight=args.weight,
        activation=args.activation,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        train_triples=train_triples,
    )
    # elif args.method == "crossentropy":
    #     kge_model = KGEV2Model(
    #         args=args,
    #         model_name=args.model,
    #         nentity=nentity,
    #         nrelation=nrelation,
    #         hidden_dim=args.hidden_dim,
    #         gamma=args.gamma,
    #         weight=args.weight,
    #         activation=args.activation,
    #         double_entity_embedding=args.double_entity_embedding,
    #         double_relation_embedding=args.double_relation_embedding,
    #         train_triples=train_triples,
    #     )  
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()

    if args.pretrained:
        path = os.path.join(args.data_path, f'RotatE_{args.hidden_dim}')
        if not os.path.exists(path):
            raise ValueError(f'Pretrained checkpoint of proposed dimension {args.hidden_dim} does not exist!')

        kge_model.entity_embedding.data = torch.from_numpy(np.load(os.path.join(path, 'entity_embedding.npy'))).cuda()
        kge_model.relation_embedding.data = torch.from_numpy(np.load(os.path.join(path, 'relation_embedding.npy'))).cuda()      
        logging.info("Loaded pretrained model from %s" % path)

    if args.pretrained_path is not None:
        kge_model.entity_embedding.data = torch.from_numpy(np.load(os.path.join(args.pretrained_path, 'entity_embedding.npy'))).cuda()
        kge_model.relation_embedding.data = torch.from_numpy(np.load(os.path.join(args.pretrained_path, 'relation_embedding.npy'))).cuda()

        model_state_dict = torch.load(os.path.join(args.pretrained_path, 'checkpoint'))
        if 'head_m_r' in model_state_dict['model_state_dict']:
            kge_model.head_m_r.data = model_state_dict['model_state_dict']['head_m_r']
            kge_model.tail_m_r.data = model_state_dict['model_state_dict']['tail_m_r']
        logging.info("Loaded pretrained model from %s" % args.pretrained_path)

    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn   # if args.method == "default" else TrainDataset.collate_fn_ce,
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn   # if args.method == "default" else TrainDataset.collate_fn_ce,
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )

        if args.training_epochs is not None:
            args.max_steps = (len(train_dataloader_head) + len(train_dataloader_tail)) * 50
        if args.evaluate_strategy == "epochs":
            args.valid_steps = (len(train_dataloader_head) + len(train_dataloader_tail)) * 5

        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2
        
        param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
        grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step
    best_valid_metrics = {}
    best_test_metrics = {}

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('max_step = %d' % args.max_steps)
    logging.info('valid_steps = %d' % args.valid_steps)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training
    train_step = kge_model.train_step
    test_step = kge_model.test_step

    if args.do_valid:
        logging.info('Evaluating on Test Dataset...')
        metrics = test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
        for k, v in metrics.items():
            if k not in best_test_metrics:
                best_test_metrics[k] = v
            else:
                if k in ["MR", "loss", 'rotate_loss', 'identity_matrix_loss']:
                    if best_test_metrics[k] > v:
                        best_test_metrics[k] = v
                else:
                    if best_test_metrics[k] < v:
                        best_test_metrics[k] = v
        log_metrics('Best-Test', step, best_test_metrics)
    
    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []
        
        #Training Loop
        for step in tqdm(range(init_step, args.max_steps)):
            
            log = train_step(kge_model, optimizer, train_iterator, args)
            
            training_logs.append(log)
            
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)
                
            if step % args.log_steps == 0:
                metrics = {}
                metrics = {'gnorm': grad_norm(kge_model), 'pnorm': param_norm(kge_model)}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            if args.do_valid and step % args.valid_steps == 0 and step != 0:
                # we temporary comment these lines to accelerate the training.
                logging.info('Evaluating on Valid Dataset...')
                metrics = test_step(kge_model, valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics)
                for k, v in metrics.items():
                    if k not in best_valid_metrics:
                        best_valid_metrics[k] = v
                    else:
                        if k == "MR":
                            if best_valid_metrics[k] > v:
                                best_valid_metrics[k] = v
                        else:
                            if best_valid_metrics[k] < v:
                                best_valid_metrics[k] = v
                log_metrics('Best-Valid', step, best_valid_metrics)

                logging.info('Evaluating on Test Dataset...')
                metrics = test_step(kge_model, test_triples, all_true_triples, args)
                log_metrics('Test', step, metrics)
                for k, v in metrics.items():
                    if k not in best_test_metrics:
                        best_test_metrics[k] = v
                    else:
                        if k in ["MR", "loss", 'rotate_loss', 'identity_matrix_loss']:
                            if best_test_metrics[k] > v:
                                best_test_metrics[k] = v
                        else:
                            if best_test_metrics[k] < v:
                                best_test_metrics[k] = v
                log_metrics('Best-Test', step, best_test_metrics)

        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)
        
    # we temporary comment these lines to accelerate the training.
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)
        for k, v in metrics.items():
            if k not in best_valid_metrics:
                best_valid_metrics[k] = v
            else:
                if k in ["MR", "loss", 'rotate_loss', 'identity_matrix_loss']:
                    if best_valid_metrics[k] > v:
                        best_valid_metrics[k] = v
                else:
                    if best_valid_metrics[k] < v:
                        best_valid_metrics[k] = v
        log_metrics('Best-Valid', step, best_valid_metrics)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
        for k, v in metrics.items():
            if k not in best_test_metrics:
                best_test_metrics[k] = v
            else:
                if k in ["MR", "loss", 'rotate_loss', 'identity_matrix_loss']:
                    if best_test_metrics[k] > v:
                        best_test_metrics[k] = v
                else:
                    if best_test_metrics[k] < v:
                        best_test_metrics[k] = v
        log_metrics('Best-Test', step, best_test_metrics)

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
        
if __name__ == '__main__':
    main(parse_args())
