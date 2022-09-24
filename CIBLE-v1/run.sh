python train.py \
    --model CIBLETransR \
    --dataset UMLS \
    --dataset_version rnnlogic \
    --batch_size 16 \
    --embedding_dim 1024 \
    --learning_rate 0.0005 \
    --regularizer none \
    --scoring_fct_norm 1