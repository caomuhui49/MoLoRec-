for name in Sports_and_Outdoors Beauty Toys_and_Games Cell_Phones_and_Accessories Clothing_Shoes_and_Jewelry Grocery_and_Gourmet_Food Health_and_Personal_Care Home_and_Kitchen Pet_Supplies Tools_and_Home_Improvement Video_Games
do
    wget -P data/rawdata http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_${name}_5.json.gz
    wget -P data/rawdata http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_${name}.json.gz
done