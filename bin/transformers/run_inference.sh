#!/usr/bin/env bash


source /home/local/QCRI/fialam/anaconda3home/bin/activate /home/local/QCRI/fialam/anaconda3home/envs/transformers


declare -a questions=("exp_q1_run_2" "exp_q2_run_9" "exp_q3_run_8" "exp_q4_run_7" "exp_q5_run_7" "exp_q6_run_5" "exp_q7_run_9" )



export RESULTS_DIR="/export/alt-tanbih/exp_covid19_disinfo/data/data_new/results"
export TASK_NAME=multiclass
export MODEL_DIR="/export/alt-tanbih/exp_covid19_disinfo/experiments/exp_bert_seed_binary"
export DATA_DIR="/export/alt-tanbih/exp_covid19_disinfo/data/data_new/"
index=1
for Q in ${questions[@]};
do
    qp_file="q"$index
    for i in `seq 0 1 9`;
    do
    fold="cv_"$i
    out_file=$qp_file"_"$fold".tsv"
    mkdir -p $DATA_DIR/$qp_file
    mkdir -p $RESULTS_DIR/$qp_file
python bin/run_inference.py \
    --model_name_or_path $MODEL_DIR/$Q/$fold/ \
    --task_name $TASK_NAME \
    --do_predict \
    --pred_file $DATA_DIR/$qp_file/$qp_file".tsv" \
    --out_file $RESULTS_DIR/$qp_file/$out_file \
    --data_dir $DATA_DIR/$qp_file/ \
    --max_seq_length 128 \
    --per_device_eval_batch_size=8   \
    --per_device_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $RESULTS_DIR/$qp_file/

	done
	index=$((index+1))
done


cd /export/alt-tanbih/my_transformers
declare -a questions=("q1" "q2" "q3" "q4" "q5" "q6" "q7" )
for Q in ${questions[@]};
do
python bin/combine_data_from_folds.py -d /export/alt-tanbih/exp_covid19_disinfo/data/data_new/results/$Q/ -o /export/alt-tanbih/exp_covid19_disinfo/data/data_new/results/$Q"_results_data.tsv"
done

declare -a questions=("q1" "q2" "q3" "q4" "q5" "q6" "q7" )
for Q in ${questions[@]};
do
python bin/compute_majority.py -i /export/alt-tanbih/exp_covid19_disinfo/data/data_new/results/$Q"_results_data.tsv" -o /export/alt-tanbih/exp_covid19_disinfo/data/data_new/results/$Q"_results_data_final.tsv"
done


declare -a questions=("q1" "q2" "q3" "q4" "q5" "q6" "q7" )

for Q in ${questions[@]};
do
    for i in `seq 0 1 9`;
    do
    echo $Q" fold_"$i
    python bin/performance_measure.py -g /export/alt-tanbih/exp_covid19_disinfo/data/data_new//$Q/$Q.tsv -c /export/alt-tanbih/exp_covid19_disinfo/data/data_new/results/$Q/$Q"_cv_"$i".tsv"
    done
done

python bin/performance_measure.py -g /export/alt-tanbih/exp_covid19_disinfo/data/data_new//q1/q1.tsv -c /export/alt-tanbih/exp_covid19_disinfo/data/data_new/results/q1/q1_cv_1.tsv


python bin/performance_measure.py -g /export/alt-tanbih/exp_covid19_disinfo/data/data_new//q1/q1.tsv -c /export/alt-tanbih/exp_covid19_disinfo/data/data_new/results/"q1_results_data_final.tsv"


python /alt/tanbih/exp_covid19_modeling/bin/combine_test_split_results.py -i /export/alt-tanbih/exp_covid19_disinfo/experiments/exp_bert_seed_binary/exp_q1_run_2/evaluation -o /export/alt-tanbih/exp_covid19_disinfo/experiments/exp_bert_seed_binary/exp_q1_run_2/evaluation/evaluation.singletest.json


python /alt/tanbih/exp_covid19_modeling/bin/combine_test_split_results.py -o /export/alt-tanbih/exp_covid19_disinfo/experiments/exp_bert_seed_binary/exp_q1_run_2/evaluation -o /export/alt-tanbih/exp_covid19_disinfo/experiments/exp_bert_seed_binary/exp_q1_run_2/evaluation/evaluation.singletest.json
/alt/tanbih/exp_covid19_modeling/bin/create_single_test_from_folds.sh ../data/binary_english_splits/q1/

bash bin/run_pr_measure.sh >/export/alt-tanbih/exp_covid19_disinfo/data/data_new/CV_results.txt