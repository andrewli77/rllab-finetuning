#!/bin/bash

for SLURM_ARRAY_TASK_ID in {0..29}
do
	idx=$SLURM_ARRAY_TASK_ID
	bs=$((idx / 15))
	idx=$((idx % 15))
	itr=$((idx / 5))
	idx=$((idx % 5))
	seed=$((idx))

	if [ $bs -eq 0 ]
	then
		bs=160000
	else
		bs=480000
	fi

	if [ $itr -eq 0 ]
	then
		itr=0
	elif [ $itr -eq 1 ]
	then
		itr=200
	fi
	
	echo "$bs, $itr, $seed"
	%python3 sandbox/finetuning/runs/pg_test.py --env_name snakegather --algo hippo -trainsnn -p 10 -tpi 10 -e 1 -n_itr 0 -eps 0.1 -bs $bs -d 0.999 -mb -sbl -msb -lr2 0 -version 1 --pkl_path data/local/$seed/itr_$itr.pkl

done