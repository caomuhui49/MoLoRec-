from transformers import AutoTokenizer
import argparse
import json
import time
from tqdm import tqdm
import torch
from mlora3.utils import load_loras, train_step
cuda_num = torch.cuda.device_count()
device = f"cuda:{cuda_num-1}"
def main(adapter_list, tta_len, max_len, batch, first_tokens, save_name, test_file, gradient_accumulation_steps, lr, func, softmax_t):
    model = load_loras(
        eval(adapter_list),
        func=func,
        softmax_t=softmax_t,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for n, p in model.named_parameters():
        if 'weights' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable weights num: {trainable_num}")
    weights = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(weights, lr=lr)
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen2-7B-Instruct", padding_side="left",use_fast=False, model_max_length=max_len)
    data = []
    with open(test_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    data = data[:tta_len]
    all_len = len(data)
    scores = ()
    epoch = [1]
    start = time.time()
    for i in tqdm(range(all_len)):
        messages = [data[i]['messages'][0], data[i]['messages'][1]]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", padding="max_length",max_length=max_len,truncation=True).to(device).input_ids
        scores = train_step(
            model.base_model.model,
            inputs,
            batch,
            first_tokens,
            scores,
            epoch,
            gradient_accumulation_steps,
            optimizer,
            max_new_tokens=400,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False
        )
    save_state = {}
    for params in model.state_dict():
        if 'weights' in params:
            save_state[params] = model.state_dict()[params]
    torch.save(save_state, save_name)
    end = time.time()
    print(f"{save_name} train Time: ", (end-start)/60/60, 'hours')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter_list', type=str)
    parser.add_argument('--tta_len', type=int)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--first_tokens', type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--func', type=str)
    parser.add_argument('--softmax_t', type=float)
    args = parser.parse_args()
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    main(
        parser.parse_args().adapter_list,
        parser.parse_args().tta_len,
        parser.parse_args().max_len, 
        parser.parse_args().batch, 
        parser.parse_args().first_tokens,
        parser.parse_args().save_name, 
        parser.parse_args().test_file, 
        parser.parse_args().gradient_accumulation_steps,
        parser.parse_args().lr,
        parser.parse_args().func,
        parser.parse_args().softmax_t
    )
