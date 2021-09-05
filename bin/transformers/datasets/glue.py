import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

# from ...tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
# from ...tokenization_utils import PreTrainedTokenizer
# from ...tokenization_xlm_roberta import XLMRobertaTokenizer
# from ..processors.glue import glue_convert_examples_to_features, glue_output_modes, glue_processors
# from ..processors.utils import InputFeatures

from transformers import RobertaTokenizer, RobertaTokenizerFast
from transformers import PreTrainedTokenizer
from transformers import XLMRobertaTokenizer
from processors.glue import glue_convert_examples_to_features,glue_convert_pred_examples_to_features, glue_output_modes, glue_processors
from processors.utils import InputFeatures


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
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    evaluation_dir: str = field(
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
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class GlueDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]



    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        evaluate_dev=False,
        evaluate_test=False,
    ):
        self.args = args
        logger.info("GLUE data parameters %s", args)
        processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        processor.set_label_index(args.label_index)
        processor.set_lang(args.lang)
        processor.set_train_file(args.train_file)
        label_list = processor.get_labels()
        logger.info("Label list %s", label_list)


        if (evaluate_dev):
            # dataset='dev'
            base_name = os.path.basename(args.dev_file)
            base_name = os.path.splitext(base_name)[0]
            dataset = base_name
        elif (evaluate_test):
            base_name = os.path.basename(args.test_file)
            base_name = os.path.splitext(base_name)[0]
            dataset = base_name
        else:
            dataset = 'train'

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                dataset, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")
                label_list = processor.get_labels()
                logger.info("Label list %s", label_list)
                if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
                    RobertaTokenizer,
                    RobertaTokenizerFast,
                    XLMRobertaTokenizer,
                ):
                    # HACK(label indices are swapped in RoBERTa pretrained model)
                    label_list[1], label_list[2] = label_list[2], label_list[1]
                # examples = (

                if evaluate_dev:
                    examples = processor.get_dev_examples(args.dev_file)
                elif evaluate_test:
                    examples = processor.get_test_examples(args.test_file)
                else:
                    examples = processor.get_train_examples(args.train_file)
                # )
                if limit_length is not None:
                    examples = examples[:limit_length]
                # print(examples)
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

class GluePredDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
        label_list= None,
        predict=False,
    ):
        self.args = args
        processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        processor.set_label_index(args.label_index)
        processor.set_lang(args.lang)
        processor.set_train_file(args.train_file)
        processor.set_labels(label_list)
        label_list = processor.get_labels()
        logger.info("Label list %s", label_list)
        # examples = (processor.get_predict_examples(args.pred_file))
        # print(examples)
        # logger.info("File %s", args.pred_file)
        # logger.info("Training/evaluation parameters %s", examples)
        # cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format("predict",
        #                                                                               list(filter(None,
        #                                                                                           args.model_name_or_path.split(
        #                                                                                               '/'))).pop(),
        #                                                                               str(args.max_seq_length),
        #                                                                               str(args.task_name)))

        base_name = os.path.basename(args.test_file)
        base_name = os.path.splitext(base_name)[0]
        dataset = base_name

        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                "predict_"+dataset, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")
                label_list = processor.get_labels()
                if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
                    RobertaTokenizer,
                    RobertaTokenizerFast,
                    XLMRobertaTokenizer,
                ):
                    # HACK(label indices are swapped in RoBERTa pretrained model)
                    label_list[1], label_list[2] = label_list[2], label_list[1]
                examples = (processor.get_test_examples(args.test_file))

                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = glue_convert_examples_to_features(  #glue_convert_pred_examples_to_features
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]