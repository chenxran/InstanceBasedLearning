# Instance-based Learning for Knowledge Base Completion

This repository is the official implementation of [Instance-based Learning for Knowledge Base Completion](https://arxiv.org/abs/2110.13577). This paper has been accepted to NeurIPS 2022. The implementation of CIBLE is based on [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).

## Abstract
In this paper, we proposed a new method for knowledge base completion (KBC): instance-based learning (IBL). For example, to answer (Jill Biden, lived city,? ), instead of going directly to Washington D.C., our goal is to find Joe Biden, who has the same lived city as Jill Biden. Through prototype entities, IBL provides interpretability. We developed theories for modeling prototypes and combining IBL with translational models. Experiments on various tasks have confirmed the IBL model’s effectiveness and interpretability.

In addition, IBL shed light on the mechanism of rule-based KBC models. Previous research has generally agreed that rule-based methods provide rules with semantically related premise and hypothesis. We challenge this view. We begin by demonstrating that some logical rules represent instance-based equivalence (i.e. prototypes) rather than semantic relevance. These are denoted as IBL rules. Surprisingly, despite occupying only a small portion of the rule space, IBL rules outperform non-IBL rules in all four benchmarks. We use a variety of experiments to demonstrate that rule-based models work because they have the ability to represent instance-based equivalence via IBL rules. The findings provide new insights of how rule-based models work and how to interpret their rules.

## Dependencies

To install requirements:

```
conda env create -f environment.yaml
conda activate cible
```

Download the datasets and corresponding checkpoints from [here](https://drive.google.com/file/d/159qfesBCgsM-MIn4MB4VV5PmiL9U9Vno/view?usp=sharing).

## Training

### Train IBLE/CIBLE
To train IBLE/CIBLE, we offer the hyper-parameter configuration and corresponding commands in run.sh. For example, you can run the following command to train IBLE on UMLS.
```
python codes/run.py --cuda --do_train --do_valid --do_test -adv --activation=tanh --batch_size=64 --cosine=True \
    --data_path=data/umls --hidden_dim=2000 --ible_weight=1 --learning_rate=0.0005 --log_steps=100 --max_steps=5000 \
    --mlp=True --model=TransE --negative_sample_size=-1 --pooling=add --regularization=0.1 --relation_aware=none \
    --save_checkpoint_steps=500 --save_path=models/umls --test_batch_size=16 --valid_steps=500
```


|   IBLE     | MR | MRR | Hit@1 | Hit@3 | Hit@10 |
|------------|-|-|-|-|-|
|   FB15k-237       | 263   | 0.284 | 20.0 | 31.0 | 45.2 |
|   WN18RR          | 7205  | 0.394 | 37.7 | 40.0 | 42.7 |
|   Kinships        | 3.7   | 0.650 | 51.3 | 75.5 | 93.7 |
|   UMLS            | 3.2   | 0.816 | 71.7 | 90.0 | 96.1 |

|   CIBLE     | MR | MRR | Hit@1 | Hit@3 | Hit@10 |
|------------|-|-|-|-|-|
|   FB15k-237       | 170   | 0.341 | 24.6 | 37.8 | 53.2 |
|   WN18RR          | 3400  | 0.490 | 44.6 | 50.7 | 57.5 |
|   Kinships        | 3.0   | 0.728 | 60.3 | 82.0 | 95.6 |
|   UMLS            | 2.6   | 0.856 | 78.7 | 91.6 | 97.0 |


### Train RNNLogic with only IBLE/non-IBLE rules
To train RNNLogic with only IBLE/non-IBLE rules, we offer the commands for corresponding tasks in RNNLogic/run.sh. For example, you can run the following command to train RNNLogic with only IBLE rules on UMLS.
```
python codes/run.py --task umls --predictor_lr 5e-5 --predictor_early_break_rate 0.2 --max_rules 1000 --max_beam_rules 1000 \
    --predictor_num_epoch 200000 --generator_num_epoch 10000 --num_em_epoch 1 --max_rule_len 3 --param_relation_embed True --train_with_rotate True \
    --train_with_pgnd True --filter_with_pgnd True --filter_with_rotate True --filter_rule True --filter_with_recall True --enumerate_symmetric_rule True \
    --only_symmetric_rule True --start 0
```
