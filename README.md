# Instance-based Learning for Knowledge Base Completion

This repository is the official implementation of [Instance-based Learning for Knowledge Base Completion](https://arxiv.org/abs/2110.13577). This paper has been accepted to NeurIPS 2022.

## Abstract
In this paper, we proposed a new method for knowledge base completion (KBC): instance-based learning (IBL). For example, to answer (Jill Biden, lived city,? ), instead of going directly to Washington D.C., our goal is to find Joe Biden, who has the same lived city as Jill Biden. Through prototype entities, IBL provides interpretability. We developed theories for modeling prototypes and combining IBL with translational models. Experiments on various tasks have confirmed the IBL model’s effectiveness and interpretability.

In addition, IBL shed light on the mechanism of rule-based KBC models. Previous research has generally agreed that rule-based methods provide rules with semantically related premise and hypothesis. We challenge this view. We begin by demonstrating that some logical rules represent instance-based equivalence (i.e. prototypes) rather than semantic relevance. These are denoted as IBL rules. Surprisingly, despite occupying only a small portion of the rule space, IBL rules outperform non-IBL rules in all four benchmarks. We use a variety of experiments to demonstrate that rule-based models work because they have the ability to represent instance-based equivalence via IBL rules. The findings provide new insights of how rule-based models work and how to interpret their rules.

## Dependencies

To install requirements:

```
conda env create -f environment.yaml
conda activate cible
```

## Training

### Train IBLE/CIBLE
To train IBLE/CIBLE, we offer the hyper-parameter configuration and corresponding commands in run.sh. For example, you can run the following command to train IBLE on UMLS.
```
python codes/run.py --cuda --do_train --do_valid --do_test -adv --activation=tanh --batch_size=64 --cosine=True \
    --data_path=data/umls --hidden_dim=2000 --ible_weight=1 --learning_rate=0.0005 --log_steps=100 --max_steps=5000 \
    --mlp=True --model=TransE --negative_sample_size=-1 --pooling=add --regularization=0.1 --relation_aware=none \
    --save_checkpoint_steps=500 --save_path=models/umls --test_batch_size=16 --valid_steps=500
```
