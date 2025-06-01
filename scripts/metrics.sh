test_file=testdata/toyswarm.jsonl
res_path=result/toyswarm_merge/output
python src/metrics.py --res_path ${res_path} --truth_file ${test_file}