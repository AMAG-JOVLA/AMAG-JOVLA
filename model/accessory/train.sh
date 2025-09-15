#!/bin/bash
module load anaconda/2020.11
module load cuda/11.8

source activate vla_model

llama_type=llama_ens5_light
pretrained_path=/HOME/scz0p9e/run/hexiangdong/vla/model/checkpoint/SPHINX-Tiny-1k
llama_config="$pretrained_path"/config.json
tokenizer_path="$pretrained_path"/tokenizer.model
data_config=configs/vla.yaml

data_parallel=sdp
model_parallel=2
lr=0.00002

exp=vla
exp_name=finetune/mm/"$exp"
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"
mkdir -p output_dir/"$exp"

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT
export MASTER_PORT=$PORT

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" \
main_finetune.py \
--output_dir output/"$exp_name" --epochs 2 --warmup_epochs 0.03 \
--batch_size 2 --accum_iter 8 --num_workers 4 \
--max_words 2048 \
--lr "$lr" --min_lr 0 --clip_grad 8 --weight_decay 0 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" --checkpointing \
--llama_type "$llama_type" --llama_config $llama_config --tokenizer_path "$tokenizer_path" \
--pretrained_path "$pretrained_path" \
--data_config $data_config --dialog \
--image_transform padded_resize --cache_ann_on_disk \
2>&1 | tee -a output/"$exp_name"/output.log


echo "exp name done: $exp_name"
