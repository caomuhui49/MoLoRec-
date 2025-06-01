for i in Sports_and_Outdoors Beauty Toys_and_Games Cell_Phones_and_Accessories Clothing_Shoes_and_Jewelry Grocery_and_Gourmet_Food Health_and_Personal_Care Home_and_Kitchen Pet_Supplies Tools_and_Home_Improvement Video_Games
do
    python data/raw_process.py --domain ${i}
    python data/data_process.py --domain ${i} --jobnum 93
done
python data/merge.py
cp -r data/processed/beauty/prompt/test.jsonl testdata/beautywarm.jsonl
cp -r data/processed/toys/prompt/test.jsonl testdata/toyswarm.jsonl
cp -r data/processed/sports/prompt/test.jsonl testdata/sportswarm.jsonl