# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments

from datasets.glue import GlueDataset,GluePredDataset
from processors.glue import glue_output_modes,glue_tasks_num_labels,glue_processors
from metrics import glue_compute_metrics
import exp_utils
import json

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    # glue_compute_metrics,
    # glue_output_modes,
    # glue_tasks_num_labels,
    set_seed,
)
import datetime

logger = logging.getLogger(__name__)

@dataclass
class GlueDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    data_dir: str = field(
        default="",
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    evaluation_dir: str = field(
        default="",
        metadata={"help": "The is results dir."}
    )
    train_file: str = field(
        default="",
        metadata={"help": "File to train"}
    )
    dev_file: str = field(
        default="",
        metadata={"help": "File to development/validation"}
    )
    test_file: str = field(
        default="",
        metadata={"help": "File to evaluation"}
    )
    pred_file: str = field(
        default="",
        metadata={"help": "File to classification/prediction"}
    )
    out_file: str = field(
        default="",
        metadata={"help": "File to output classification/prediction"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    label_index: int = field(
        default=2,
        metadata={"help": "label index"},
    )
    lang: str = field(
        default="english",
        metadata={"help": "select language"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

def get_results(trainer,test_datasets,label_list):

    label_map = {i: label for i, label in enumerate(label_list)}
    # results = {}
    probabilities = []
    predictions_list=[]
    gold_labels_all = []
    # if training_args.do_predict:
    for test_dataset in test_datasets:
        result = trainer.predict(test_dataset=test_dataset)
        logits=torch.from_numpy(result.predictions)
        logits= F.softmax(logits,dim=1)
        logits=logits.detach().cpu().numpy() #logits.cpu().numpy()
        predictions_index = np.argmax(logits, axis=1)
        gold_labels=result.label_ids #.detach().cpu().numpy()
        # for index, item in enumerate(result.predictions):
        # item = test_dataset.get_labels()[item]
        gold_labels = result.label_ids
        for index, item in enumerate(gold_labels):
            gold_labels_all.append(label_map[item])

        for index in predictions_index:
            label=label_map[index]
            predictions_list.append(label)
        #predictions_list.update(predictions_index)
        for index, prob in zip(predictions_index, logits):
            probabilities.append(prob[index])
    return predictions_list, probabilities,gold_labels_all,logits

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, GlueDataTrainingArguments, TrainingArguments))


    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        # num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
        processor = glue_processors[data_args.task_name]()
        processor.set_label_index(data_args.label_index)
        processor.set_lang(data_args.lang)
        processor.set_train_file(data_args.train_file)

        label_list = processor.get_labels()
        logger.info("label_list " + str(label_list))
        num_labels = len(label_list)

    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))



    logger.info("\n\nData parameters %s", data_args)
    logger.info("\n\nModel parameters %s", model_args)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )




    # Get datasets
    train_dataset = GlueDataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, evaluate_dev=True,evaluate_test=False) # if training_args.do_eval else None


    def compute_metrics(p: EvalPrediction) -> Dict:
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

    # Initialize our Trainer
    # model.resize_token_embeddings(len(tokenizer))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        # if data_args.task_name == "mnli":
        #     mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
        #     eval_datasets.append(GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, evaluate=True))

        for eval_dataset in eval_datasets:
            result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            results.update(result)

    test_datasets = [train_dataset]

    train_predictions, train_probabilities, train_gold_labels, train_logits=get_results(trainer,test_datasets,label_list)

    test_datasets = [eval_dataset]
    dev_predictions, dev_probabilities, dev_gold_labels, dev_logits=get_results(trainer,test_datasets,label_list)

    # data_args.pred_file=data_args.test_file
    # data_args.dev_file = data_args.test_file
    # test_dataset = GluePredDataset(data_args, tokenizer=tokenizer, predict=True)
    # test_dataset = GlueDataset(data_args, tokenizer=tokenizer, evaluate_dev=False, evaluate_test=True)
    # data_args.dev_file=data_args.test_file
    # test_dataset = GlueDataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None

    test_dataset = GlueDataset(data_args, tokenizer=tokenizer, evaluate_test=True) if training_args.do_eval else None
    test_datasets = [test_dataset]
    test_predictions, test_probabilities, test_gold_labels, test_logits=get_results(trainer,test_datasets,label_list)

    print()
    print("=================================")
    overall_train_results = exp_utils.compute_aggregate_scores(
        train_gold_labels, train_predictions, label_list
    )
    overall_dev_results = exp_utils.compute_aggregate_scores(
        dev_gold_labels,dev_predictions, label_list
    )
    overall_test_results = exp_utils.compute_aggregate_scores(
        test_gold_labels, test_predictions, label_list
    )
    print("Accuracy: %0.3f" % (overall_test_results["accuracy"]))
    # print("Micro Precision: %0.3f" % (overall_test_results["prf_micro"][0]))
    # print("Micro Recall: %0.3f" % (overall_test_results["prf_micro"][1]))
    print("Micro F1: %0.3f" % (overall_test_results["prf_weighted"][2]))
    print("Confusion Matrix")
    print(overall_test_results["confusion_matrix"])

    output_eval_file = os.path.join(
        data_args.out_file
    )
    #os.path.join(data_args.evaluation_dir, "evaluation.overall.json")
    with open(
            output_eval_file, "w"
    ) as eval_file:
        json.dump(exp_utils.convert_to_json({
            'train': {
                "results":overall_train_results,
                 "predictions": train_predictions,
                 "probabilities": train_probabilities,
                 "gold_labels":train_gold_labels,
                 "logits":train_logits
            },

            'dev': {"results":overall_dev_results,
                    "predictions": dev_predictions,
                    "probabilities": dev_probabilities,
                    "gold_labels": dev_gold_labels,
                    "logits": dev_logits
                    },
            'test': {
                     "results":overall_test_results,
                     "predictions": test_predictions,
                     "probabilities": test_probabilities,
                     "gold_labels": test_gold_labels,
                     "logits": test_logits
                     }
        }), eval_file)

    # logger.info("Results %s", predictions_list)
    # logger.info("Results %s", probabilities)
    # out_file_name=data_args.out_file

    # return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    a = datetime.datetime.now().replace(microsecond=0)
    main()
    b = datetime.datetime.now().replace(microsecond=0)
    print ("time taken:\t{}".format((b - a)))
