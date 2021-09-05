
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
    test_file: str = field(
        default="",
        metadata={"help": "File to evaluation"}
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

        for index in predictions_index:
            label=label_map[index]
            predictions_list.append(label)

        for index, prob in zip(predictions_index, logits):
            probabilities.append(prob[index])
    return predictions_list, probabilities #,gold_labels_all

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    parser = HfArgumentParser((ModelArguments, GlueDataTrainingArguments, TrainingArguments))
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        # print(data_args)

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
        num_labels = len(label_list)

    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    print("cache dir: {}".format(model_args.cache_dir))
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
    train_gold_labels = exp_utils.read_labels(data_args.train_file, data_args.label_index)

    label_list = list(set(train_gold_labels))
    label_list.sort()

    label_map = {i: label for i, label in enumerate(label_list)}
    print("label_map {}".format(label_map))

    test_dataset = GluePredDataset(data_args, tokenizer=tokenizer,label_list=label_list,predict=True)
    # test_dataset = GlueDataset(data_args, tokenizer=tokenizer, evaluate_dev=True, evaluate_test=False)

    # # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=None,
    )


    test_datasets = [test_dataset]
    test_predictions, test_probabilities=get_results(trainer,test_datasets,label_list)
    test_gold_labels  = exp_utils.read_labels(data_args.test_file, data_args.label_index)

    print()
    print("=================================")
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
            'test': {
                     "results":overall_test_results,
                     "predictions": test_predictions,
                     "probabilities": test_probabilities,
                     "gold_labels": test_gold_labels
                     }
        }), eval_file)

            # results.update(result)

    # print(results)
    # return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
