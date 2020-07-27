#!/bin/bash

bs=$1

for itr in 0 200 1000
do
	for seed in {0..11}
	do
		echo "python3 sandbox/finetuning/runs/pg_test.py --env_name snakegather --algo hippo -trainsnn -p 10 -tpi 10 -e 1 -n_itr 0 -eps 0.1 -bs $bs -d 0.999 -mb -sbl -msb -lr2 0 -version 1 --pkl_path data/local/good/$seed/itr_$itr.pkl"
	done
done
