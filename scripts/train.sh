
GPU_NUM=1
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29512


DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

PY_ARGS=${@:1}  # Any other arguments 

torchrun $DISTRIBUTED_ARGS main_finetune.py \
    --model AIDE \
    --resnet_path 'None' \
    --batch_size 32 \
    --blr 1e-4 \
    --epochs 10 \
    ${PY_ARGS}
