# Run RNNLogic for different settings

## Kinship
### Only use IBLE rules
python codes/run.py --task kinship --predictor_lr 5e-5 --predictor_early_break_rate 0.2 --max_rules 1000 --max_beam_rules 1000 \
    --predictor_num_epoch 200000 --generator_num_epoch 10000 --num_em_epoch 1 --max_rule_len 3 --param_relation_embed True --train_with_rotate True \
    --train_with_pgnd True --filter_with_pgnd True --filter_with_rotate True --filter_rule True --filter_with_recall True --enumerate_symmetric_rule True \
    --only_symmetric_rule True --start 0

### Only use non-IBLE rules
python codes/run.py --task kinship --predictor_lr 5e-5 --predictor_early_break_rate 0.2 --max_rules 1000 --max_beam_rules 1000 \
    --predictor_num_epoch 200000 --generator_num_epoch 10000 --num_em_epoch 1 --max_rule_len 3 --param_relation_embed True --train_with_rotate True \
    --train_with_pgnd True --filter_with_pgnd True --filter_with_rotate True --filter_rule True --filter_with_recall True --without_symmetric_rule True --start 0


## UMLS
### Only use IBLE rules
python codes/run.py --task umls --predictor_lr 5e-5 --predictor_early_break_rate 0.2 --max_rules 1000 --max_beam_rules 1000 \
    --predictor_num_epoch 200000 --generator_num_epoch 10000 --num_em_epoch 1 --max_rule_len 3 --param_relation_embed True --train_with_rotate True \
    --train_with_pgnd True --filter_with_pgnd True --filter_with_rotate True --filter_rule True --filter_with_recall True --enumerate_symmetric_rule True \
    --only_symmetric_rule True --start 0

### Only use non-IBLE rules
python codes/run.py --task umls --predictor_lr 5e-5 --predictor_early_break_rate 0.2 --max_rules 1000 --max_beam_rules 1000 \
    --predictor_num_epoch 200000 --generator_num_epoch 10000 --num_em_epoch 1 --max_rule_len 3 --param_relation_embed True --train_with_rotate True \
    --train_with_pgnd True --filter_with_pgnd True --filter_with_rotate True --filter_rule True --filter_with_recall True --without_symmetric_rule True --start 0


## FB15k-237
### Only use IBLE rules
python codes/run.py --task FB15k-237 --predictor_lr 5e-5 --predictor_early_break_rate 0.2 --max_rules 1000 --max_beam_rules 1000 \
    --predictor_num_epoch 200000 --generator_num_epoch 10000 --num_em_epoch 1 --max_rule_len 3 --param_relation_embed True --train_with_rotate True \
    --train_with_pgnd True --filter_with_pgnd True --filter_with_rotate True --filter_rule True --filter_with_recall True --enumerate_symmetric_rule True \
    --only_symmetric_rule True --start 0

### Only use non-IBLE rules
python codes/run.py --task FB15k-237 --predictor_lr 5e-5 --predictor_early_break_rate 0.2 --max_rules 1000 --max_beam_rules 1000 \
    --predictor_num_epoch 200000 --generator_num_epoch 10000 --num_em_epoch 1 --max_rule_len 3 --param_relation_embed True --train_with_rotate True \
    --train_with_pgnd True --filter_with_pgnd True --filter_with_rotate True --filter_rule True --filter_with_recall True --without_symmetric_rule True --start 0


## WN18RR
### Only use IBLE rules
python codes/run.py --task wn18rr --predictor_lr 5e-5 --predictor_early_break_rate 0.2 --max_rules 1000 --max_beam_rules 1000 \
    --predictor_num_epoch 200000 --generator_num_epoch 10000 --num_em_epoch 1 --max_rule_len 3 --param_relation_embed True --train_with_rotate True \
    --train_with_pgnd True --filter_with_pgnd True --filter_with_rotate True --filter_rule True --filter_with_recall True --enumerate_symmetric_rule True \
    --only_symmetric_rule True --start 0

### Only use non-IBLE rules
python codes/run.py --task wn18rr --predictor_lr 5e-5 --predictor_early_break_rate 0.2 --max_rules 1000 --max_beam_rules 1000 \
    --predictor_num_epoch 200000 --generator_num_epoch 10000 --num_em_epoch 1 --max_rule_len 3 --param_relation_embed True --train_with_rotate True \
    --train_with_pgnd True --filter_with_pgnd True --filter_with_rotate True --filter_rule True --filter_with_recall True --without_symmetric_rule True --start 0