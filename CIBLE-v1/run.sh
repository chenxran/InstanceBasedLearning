CUDA_VISIBLE_DEVICES=0 python train.py \
--batch_size=128 \
--dataset=Kinships \
--dataset_version=rnnlogic \
--embedding_dim=1024 \
--epochs=50 \
--learning_rate=0.0005 \
--mlp=True \
--model=mf-transr \
--regularizer=none \
--scoring_fct_norm=2 \
--transe_weight=0.1



    