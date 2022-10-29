# Running IBLE on UMLS 
python codes/run.py --cuda --do_train --do_valid --do_test -adv --activation=tanh --batch_size=64 --cosine=True \
    --data_path=data/umls --hidden_dim=2000 --ible_weight=1 --learning_rate=0.0005 --log_steps=100 --max_steps=5000 \
    --mlp=True --model=TransE --negative_sample_size=-1 --pooling=add --regularization=0.1 --relation_aware=none \
    --save_checkpoint_steps=500 --save_path=models/umls --test_batch_size=16 --valid_steps=500

# Running IBLE on Kinship
python codes/run.py --cuda --do_train --do_valid --do_test -adv --activation=tanh --batch_size=64 --cosine=True \
    --data_path=data/kinship --hidden_dim=1000 --ible_weight=1 --learning_rate=0.0005 --log_steps=100 --max_steps=5000 \
    --mlp=True --model=TransE --negative_sample_size=-1 --pooling=add --regularization=0.2 --relation_aware=none \
    --save_checkpoint_steps=500 --save_path=models/kinship --test_batch_size=16 --valid_steps=500


# Running CIBLE on UMLS
python codes/run.py --cuda --do_train --do_valid --do_test -de -adv --pretrained --batch_size=256 --data_path=data/umls --gamma=12 \
    --hidden_dim=2000 --ible_weight=0.05 --learning_rate=0.0002 --log_steps=100 --max_steps=6000 --model=RotatE --negative_sample_size=-1 \
    --regularization=0.2 --save_checkpoint_steps=500 --save_path=models/umls --test_batch_size=16 --valid_steps=500

# Running CIBLE on Kinship
python codes/run.py --cuda --do_train --do_valid --do_test -de -adv --pretrained --batch_size=512 --data_path=data/kinship --gamma=15 \
    --hidden_dim=2000 --ible_weight=0.05 --learning_rate=0.0005 --log_steps=100 --max_steps=6000 --model=RotatE --negative_sample_size=-1 \
    --regularization=0.2 --save_checkpoint_steps=500 --save_path=models/kinship --test_batch_size=16 --valid_steps=500

# Running CIBLE on wn18rr
python codes/run.py --cuda --do_train --do_valid --do_test -de -adv --pretrained --activation=tanh --adversarial_temperature=0.5 \
    --batch_size=8 --cosine=True --data_path=data/wn18rr --gamma=9 --gradient_accumulation_steps=64 --hidden_dim=500 --ible_weight=0.1 \
    --learning_rate=5e-05 --log_steps=100 --loss=margin --max_steps=5000 --model=RotatE --negative_sample_size=16 --pooling=add \
    --relation_aware=True --save_checkpoint_steps=500 --save_path=models/wn18rr --test_batch_size=16 --valid_steps=500

# Running CIBLE on FB15k-237
python codes/run.py --cuda --do_train --do_valid --do_test -de -adv --pretrained --adversarial_temperature=2 --batch_size=16 --cosine=True \
    --data_path=data/FB15k-237 --gamma=15 --gradient_accumulation_steps=64 --hidden_dim=1000 --ible_weight=0.05 --learning_rate=1e-05 \
    --log_steps=100 --loss=margin --max_steps=5000 --mlp=True --model=RotatE --negative_sample_size=64 --pooling=add --relation_aware=True \
    --save_checkpoint_steps=500 --save_path=models/RotatE_umls_0 --test_batch_size=16 --valid_steps=500
