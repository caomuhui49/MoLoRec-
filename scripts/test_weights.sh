adapter_list="['save/amazon/checkpoint-1230','save/toys/checkpoint-1140']"
test_file="testdata/toyswarm.jsonl"
output=result/toyswarm
func="softmax"
softmax_t=1
out=${output}/output
if [ ! -d ${out} ];then
    mkdir ${out}
fi
for lr in 5e-1 6e-1 7e-1 8e-1 9e-1 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=0 python src/merge_weights.py \
        --adapter_list ${adapter_list} \
        --weight_path ${output}/lr${lr}_${func}.pth \
        --func ${func} \
        --softmax_t ${softmax_t}
    CUDA_VISIBLE_DEVICES=0,1,2,3 python src/vllmtest.py \
        --outpath ${out} \
        --outname lr${lr}_${func} \
        --input ${test_file} \
        --memory 1 \
        --batch 350 \
        --max_new_tokens 800 \
        --model ${output}/lr${lr}_${func}
    rm -rf ${output}/lr${lr}_${func}
done