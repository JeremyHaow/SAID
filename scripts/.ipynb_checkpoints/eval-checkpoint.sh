GPU_NUM=1
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29572

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
RESUME_PATH="./results/AIDE_ghe_score_patch_128/train_ForenSynths/20250529_233604"

eval_datasets=(
    "/root/lanyun-tmp/datasets/AIGCDetectionBenchMark/test" \
    "/root/lanyun-tmp/datasets/GenImage/test" \
    "/root/lanyun-tmp/datasets/ForenSynths/test" \
    "/root/lanyun-tmp/datasets/UniversalFakeDetect" \
    "/root/lanyun-tmp/datasets/Self-Synthesis/test" \
    "/root/lanyun-tmp/datasets/Chameleon/test"
)

for eval_dataset in "${eval_datasets[@]}"
do
    torchrun $DISTRIBUTED_ARGS main_finetune.py \
        --model AIDE \
        --resnet_path 'None' \
        --batch_size 32 \
        --data_path "$eval_dataset" \
        --eval_data_path "$eval_dataset" \
        --output_dir "$RESUME_PATH/eval" \
        --log_dir "$RESUME_PATH/eval/log" \
        --eval True \
        --resume "$RESUME_PATH/checkpoint-best.pth"
done