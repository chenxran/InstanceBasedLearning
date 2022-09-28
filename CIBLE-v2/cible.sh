task=$1
gpu_id=$2

if [[ $task = "kinships" ]]; then
    CUDA_VISIBLE_DEVICES=$gpu_id python codes/run.py 
        --cuda \
        --do_train \
        --do_valid \
        --do_test -de -adv \
        --adversarial_temperature=0.5 \
        --batch_size=256 \
        --data_path=data/kinship \
        --gamma=15 \
        --hidden_dim=2000 \
        --learning_rate=0.0005 \
        --log_steps=100 \
        --max_steps=6000 \
        --model=CIBLERotatE \
        --negative_sample_size=64 \
        --pretrained=True \
        --save_checkpoint_steps=500 \
        --save_path=models/RotatE_umls_0 \
        --temperature=1 \
        --test_batch_size=16 \
        --valid_steps=500 \
        --weight=0.95
elif [[ $task == "umls" ]]; then
    CUDA_VISIBLE_DEVICES=$gpu_id python codes/run.py --cuda --do_train --do_valid --do_test -de -adv --adversarial_temperature=2 --batch_size=128 --data_path=data/umls --gamma=12 --hidden_dim=2000 --learning_rate=0.0005 --log_steps=100 --max_steps=6000 --model=CIBLERotatE --negative_sample_size=32 --pooling=mean --pretrained=True --save_checkpoint_steps=500 --save_path=models/RotatE_umls_0 --temperature=1 --test_batch_size=16 --valid_steps=500 --weight=0.9
