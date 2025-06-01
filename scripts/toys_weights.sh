adapter_list="['save/amazon/checkpoint-1230','save/toys/checkpoint-1140']"
test_file="testdata/toyswarm.jsonl"
bs=60
gs=4
tta_len=100
func="softmax"
first_tokens=3
softmax_t=1
output=result/toyswarm
if [ ! -d ${output} ];then
    mkdir ${output}
fi
for lr in 5e-1 6e-1 7e-1 8e-1 9e-1 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=0,1 python -u src/train_weights.py \
        --adapter_list ${adapter_list} \
        --tta_len ${tta_len} \
        --max_len 1200 \
        --batch ${bs} \
        --gradient_accumulation_steps ${gs} \
        --lr ${lr} \
        --func ${func} \
        --softmax_t ${softmax_t} \
        --first_tokens ${first_tokens} \
        --save_name ${output}/lr${lr}_${func}.pth \
        --test_file ${test_file} \
        2>&1 | tee -a ${output}/lr${lr}_${func}.log
done
test_file="testdata/toyscold.jsonl"
output=result/toyscold
tta_len=50
if [ ! -d ${output} ];then
    mkdir ${output}
fi
for lr in 1e-2 2e-2 3e-2 4e-2 5e-2 6e-2 7e-2 8e-2 9e-2
do
    CUDA_VISIBLE_DEVICES=0,1 python -u src/train_weights.py \
        --adapter_list ${adapter_list} \
        --tta_len ${tta_len} \
        --max_len 1200 \
        --batch ${bs} \
        --gradient_accumulation_steps ${gs} \
        --lr ${lr} \
        --func ${func} \
        --softmax_t ${softmax_t} \
        --first_tokens ${first_tokens} \
        --save_name ${output}/lr${lr}_${func}.pth \
        --test_file ${test_file} \
        2>&1 | tee -a ${output}/lr${lr}_${func}.log
done