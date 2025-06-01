import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--domain', type=str)
domain = parser.parse_args().domain
small = domain.split('_')[0].lower()

review = pd.read_json(f'data/rawdata/reviews_{domain}_5.json', lines=True)
item_ids, user_ids = [], []
with open(f'{small}/data/item_ids.jsonl', 'r') as f:
    for line in f:
        item_ids.append(json.loads(line))
with open(f'{small}/data/user_ids.jsonl', 'r') as f:
    for line in f:
        user_ids.append(json.loads(line))
print(f'{small}: Users: {len(user_ids)}, Items: {len(item_ids)}, Interaction: {len(review)}, Density: {(len(review)/(len(user_ids)*len(item_ids)))*100}')