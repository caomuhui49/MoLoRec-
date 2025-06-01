import json
import math
import argparse
import os

def score(rank, truth, k):
    len_t = len(truth)
    hr = 0
    ndcg = 0
    for i in range(len(rank)):
        for j in range(min(k, len(rank[i]))):
            if truth[i] == rank[i][j]:
                hr += 1
                ndcg += 1 / math.log2(j+2)
    hr /= len_t
    ndcg /= len_t
    return hr, ndcg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth_file', type=str)
    parser.add_argument('--res_path', type=str)
    truth = []
    with open(parser.parse_args().truth_file, 'r') as f:
        for line in f:
            truth.append(json.loads(line))
    hn = dict()
    for file in os.listdir(parser.parse_args().res_path):
        res = []
        with open(os.path.join(parser.parse_args().res_path, file), 'r') as f:
            for line in f:
                res.append(json.loads(line))
        t = []
        for i in truth:
            t.append(i['messages'][2]['content'].split('||')[0])
        r = []
        for i in res:
            r.append(i.split('||'))
        hr_1, ndcg_1 = score(r, t, 1)
        hr_3, ndcg_3 = score(r, t, 3)
        hn[file] = [hr_1, hr_3, ndcg_3]
    best = max(hn, key=lambda x: hn[x][0]+hn[x][1]+hn[x][2])
    tail = best.find(".jsonl")
    best_model = best[:tail]
    print('best model:', best_model)
    print('hr@1:', hn[best][0])
    print('hr@3:', hn[best][1], 'ndcg@3:', hn[best][2])
    # path = parser.parse_args().res_path.split('/')[:-1]
    # path.append(best_model)
    # path = '/'.join(path)+'.pth'
    # import torch
    # weights = torch.load(path, map_location='cpu', weights_only=True)
    # w = torch.softmax(torch.cat([weights[k].unsqueeze(0) for k in weights.keys()], dim=0), dim=-1)
    # mean = w.mean(dim=0)
    # print(mean)