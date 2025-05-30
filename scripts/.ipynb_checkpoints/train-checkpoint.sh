GPU_NUM=1
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12588

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

train_datasets=(
    "/root/lanyun-tmp/datasets/train_ForenSynths/train"
)
eval_datasets=(
    "/root/lanyun-tmp/datasets/train_ForenSynths/val"
)

MODEL_NAME="AIDE"
EXPERIMENT_TAG="ghe_score_patch_128"

PY_ARGS=${@:1}

for train_dataset in "${train_datasets[@]}"
do
    for eval_dataset in "${eval_datasets[@]}"
    do
        current_time=$(date +"%Y%m%d_%H%M%S")
        train_dataset_name=$(basename "$train_dataset")
        eval_dataset_name=$(basename "$eval_dataset")
        
        OUTPUT_PATH="results/${MODEL_NAME}_${EXPERIMENT_TAG}/train_ForenSynths/${current_time}"
        
        echo "-----------------------------------------------------------------------"
        echo "Starting training run:"
        echo "  Train Dataset: $train_dataset"
        echo "  Eval Dataset:  $eval_dataset"
        echo "  Output Path:   $OUTPUT_PATH"
        echo "-----------------------------------------------------------------------"
        
        mkdir -p "$OUTPUT_PATH"

        torchrun $DISTRIBUTED_ARGS main_finetune.py \
            --model AIDE \
            --resnet_path 'None' \
            --data_path "$train_dataset" \
            --eval_data_path "$eval_dataset" \
            --batch_size 32 \
            --blr 1e-2 \
            --epochs 20 \
            --warmup_epochs 1 \
            --weight_decay 0.01 \
            --save_ckpt_freq 1 \
            --num_workers 16 \
            --output_dir "$OUTPUT_PATH" \
            --log_dir "$OUTPUT_PATH"/tensorboard_log \
            ${PY_ARGS} 2>&1 | tee -a "$OUTPUT_PATH/log_train.txt"
    done
done