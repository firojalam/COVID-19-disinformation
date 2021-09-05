#! /bin/bash

source your_path/bin/activate your_path/envs/transformers

HOME_DIR="/your_path/exp_covid19_disinfo"


train=$HOME_DIR/"data/english/covid19_disinfo_english_multiclass_train.tsv"
dev=$HOME_DIR/"data/english/covid19_disinfo_english_multiclass_dev.tsv"
tst=$HOME_DIR/"data/english/covid19_disinfo_english_multiclass_test.tsv"
task="multiclass"

export GLUE_DIR=data
export TASK_NAME=multiclass

fname=`basename $train .tsv`
tst_fname=`basename $tst .tsv`

declare -a questions=("q1" "q2" "q3" "q4" "q5" "q6" "q7" )
label_index=1

for q_dir in ${questions[@]};
do
	label_index=$((label_index+1))
	echo $label_index

	if [ $task == "multiclass" ] && [ $q_dir == "q1" ]
	then
	    echo $task
	    continue
	fi

	pretrained_model="/your_path/crawl-300d-2M-subword.vec"

    label_index=$((label_index+1))
    echo $label_index
    fname=`basename $train .tsv`

    exp_dir=${HOME_DIR}/exp_fasttext/en/$q_dir/$fname/
    mkdir -p ${exp_dir}
    python bin/fasttext_exp.py --train-file $train  --dev-file $dev  --test-file $tst  --exp-dir $exp_dir --label_index $label_index --lang $lang --dim 300 --pretrainedVectors $pretrained_model --thread 20 --autotuneDuration 0
done
