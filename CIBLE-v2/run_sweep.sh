gpu_id=$1
task=$2
# Kinships

if [[ $task == "kinship" ]]; then
    CUDA_VISIBLE_DEVICES=$gpu_id wandb agent chenxran/cible/p0bgnson
elif [[ $task == "umls" ]]; then
    CUDA_VISIBLE_DEVICES=$gpu_id wandb agent chenxran/cible/e2ldjodi
fi
