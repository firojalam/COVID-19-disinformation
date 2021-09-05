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

  seed=42
  run="run_"$seed
  output_file=$HOME_DIR/results/bert/en/$q_dir/$fname/$tst_fname"_"$run".json"
  mkdir -p $HOME_DIR/results/bert/en/$q_dir/$fname/

  export GLUE_DIR=cache/
  export TASK_NAME=multiclass
  outputdir=$HOME_DIR/experiments/bert/en/$q_dir/$fname/$run/
  data_dir=$GLUE_DIR/$TASK_NAME/bert/en/$q_dir/$fname/$run/
  mkdir -p $outputdir
  mkdir -p $data_dir
  python bin/transformers/run_glue.py \
      --model_name_or_path bert-base-uncased \
      --label_index $label_index \
      --task_name $TASK_NAME \
      --train_file $train \
      --dev_file $dev \
      --test_file $tst \
      --out_file $output_file \
      --do_train \
      --do_eval \
      --data_dir $data_dir \
      --max_seq_length 128 \
      --per_device_eval_batch_size=32   \
      --per_device_train_batch_size=32   \
      --learning_rate 2e-5 \
      --num_train_epochs 10.0 \
      --output_dir $outputdir \
      --seed $seed \
      --overwrite_output_dir

done
