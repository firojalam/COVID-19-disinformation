import csv
import glob
import os
import json

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as compute_confusion_matrix,
    precision_recall_fscore_support,
)

import numpy as np

# Returns all the rows in the tsv (ignoring header row) as a list of tuples, where
# each element in the list is an (instance, label) pair corresponding to a line in
# the tsv
def load_tsv(tsv_path, data_column, label_column):
    data = []
    with open(tsv_path) as tsvfile:
        input_file = csv.reader(tsvfile, delimiter="\t", quotechar="")
        for row_idx, row in enumerate(input_file):
            if row_idx == 0:
                # Ignore header row
                continue
            data.append((row[data_column], row[label_column]))
    return data


# Loads all the datasplits and returns an array of folds, where each element is
# a 3-tuple with (train, dev, test) data. Each dataset is represented as a list of
# (instance, label) tuples
def load_data_splits(base_dir):
    num_folds = len(glob.glob(os.path.join(base_dir, "train*")))

    assert num_folds == len(
        glob.glob(os.path.join(base_dir, "dev*"))
    ), "Number of dev folds not the same as train folds"
    assert num_folds == len(
        glob.glob(os.path.join(base_dir, "test*"))
    ), "Number of test folds not the same as train folds"

    folds = [None] * num_folds

    for train_file_path in glob.glob(os.path.join(base_dir, "train*")):
        train_file_name = train_file_path.split("/")[-1]
        fold_idx = int(train_file_name.split(".")[1])

        dev_file_name = "dev.%d.tsv" % (fold_idx)
        test_file_name = "test.%d.tsv" % (fold_idx)

        folds[fold_idx] = (
            load_tsv(os.path.join(base_dir, train_file_name), 2, 3),
            load_tsv(os.path.join(base_dir, dev_file_name), 2, 3),
            load_tsv(os.path.join(base_dir, test_file_name), 2, 3),
        )

        print("Loaded Fold %02d" % (fold_idx))
        print(
            "Instances: %d train, %d dev, %d test"
            % (
                len(folds[fold_idx][0]),
                len(folds[fold_idx][1]),
                len(folds[fold_idx][2]),
            )
        )

    return folds


# Create experiment directory and subdirectories for saving models, evaluations
# and temporary files
def prepare_experiment_dirs(exp_dir):
    # Prepare experiment directories
    exp_dir, tmp_dir, models_dir, evaluation_dir = get_experiment_dirs(exp_dir)

    for directory in [exp_dir, tmp_dir, models_dir, evaluation_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)


# Get subdirectory paths
def get_experiment_dirs(exp_dir):
    return (
        exp_dir,
        os.path.join(exp_dir, "tmp"),
        os.path.join(exp_dir, "models"),
        os.path.join(exp_dir, "evaluation"),
    )


# Convert a dictionary to JSON-able object, converting all numpy arrays to python
# lists
def convert_to_json(obj):
    converted_obj = {}

    for key, value in obj.items():
        if isinstance(value, dict):
            converted_obj[key] = convert_to_json(value)
        else:
            # print(key, type(value))
            converted_obj[key] = np.array(value).tolist()
            # getattr(value, "tolist", lambda: value)()
            # print(key, type(converted_obj[key]))

    return converted_obj


# Utility function to compute precision, recall and F1 from a confusion matrix
def cm_to_precision_recall_f1(cm, all_classes):
    assert len(all_classes) == cm.shape[0]
    num_classes = len(all_classes)
    constructed_y_true = []
    constructed_y_pred = []
    for class_idx in range(num_classes):
        constructed_y_true.extend([all_classes[class_idx]] * np.sum(cm[class_idx, :]))
        for pred_class_idx in range(num_classes):
            constructed_y_pred.extend(
                [all_classes[pred_class_idx]] * cm[class_idx, pred_class_idx]
            )

    constructed_y_true = np.array(constructed_y_true)
    constructed_y_pred = np.array(constructed_y_pred)

    return precision_recall_fscore_support(
        constructed_y_true, constructed_y_pred, labels=all_classes, average="weighted"
    )[:-1]


# Function to compute aggregated scores from per-fold evaluations
def compute_aggregate_scores(evaluations, all_classes):
    # Collect all predictions
    all_labels = []
    all_predictions = []
    for labels, predictions in evaluations:
        all_labels += labels
        all_predictions += predictions

    aggregated_metrics = {}

    accuracy = accuracy_score(all_labels, all_predictions)
    confusion_matrix = compute_confusion_matrix(all_labels, all_predictions, labels=all_classes)
    prf_per_class = precision_recall_fscore_support(
            all_labels, all_predictions, labels=all_classes, average=None
        )[:-1]
    prf_micro = precision_recall_fscore_support(
            all_labels, all_predictions, labels=all_classes, average='micro'
        )[:-1]
    prf_macro = precision_recall_fscore_support(
            all_labels, all_predictions, labels=all_classes, average='macro'
        )[:-1]
    prf_weighted = precision_recall_fscore_support(
            all_labels, all_predictions, labels=all_classes, average='weighted'
        )[:-1]
    aggregated_metrics = {
        "accuracy": accuracy,
        "prf_per_class": prf_per_class,
        "prf_per_class_labels": all_classes,
        "prf_micro": prf_micro,
        "prf_macro": prf_macro,
        "prf_weighted": prf_weighted,
        "confusion_matrix": confusion_matrix,
    }

    return aggregated_metrics


# Function to evaluate a model on a given fold
def evaluate_model(model, fold, predict_func, exp_dir, all_classes, **kwargs):
    instances = [instance for instance, _ in fold]
    labels = [label for _, label in fold]

    predictions, probabilities = predict_func(model, instances, exp_dir, **kwargs)

    print("Computing Scores")
    accuracy = accuracy_score(labels, predictions)
    confusion_matrix = compute_confusion_matrix(labels, predictions, labels=all_classes)
    prf_per_class = precision_recall_fscore_support(
            labels, predictions, labels=all_classes, average=None
        )[:-1]
    prf_micro = precision_recall_fscore_support(
            labels, predictions, labels=all_classes, average='micro'
        )[:-1]
    prf_macro = precision_recall_fscore_support(
            labels, predictions, labels=all_classes, average='macro'
        )[:-1]
    prf_weighted = precision_recall_fscore_support(
            labels, predictions, labels=all_classes, average='weighted'
        )[:-1]
    scores = {
        "accuracy": accuracy,
        "prf_per_class": prf_per_class,
        "prf_per_class_labels": all_classes,
        "prf_micro": prf_micro,
        "prf_macro": prf_macro,
        "prf_weighted": prf_weighted,
        "confusion_matrix": confusion_matrix,
    }

    return predictions, probabilities, scores


# Function to run model training and evaluation pipeline
def run_model(exp_dir, folds, train_func, predict_func, **kwargs):
    print("Preparing model")
    num_folds = len(folds)
    exp_dir, tmp_dir, models_dir, evaluation_dir = get_experiment_dirs(exp_dir)

    # Compute list of all classes since some folds might be missing some classes
    # and this affects the metric computation like confusion matrices
    all_classes = set()
    for train_fold, dev_fold, test_fold in folds:
        all_classes.update([label for _, label in train_fold])
        all_classes.update([label for _, label in dev_fold])
        all_classes.update([label for _, label in test_fold])
    all_classes = sorted(list(all_classes))

    models = [None] * num_folds
    all_train_evaluations = []
    all_dev_evaluations = []
    all_test_evaluations = []

    for fold_idx in range(num_folds):
        print("Training fold %d" % (fold_idx))
        model = train_func(folds[fold_idx][0], folds[fold_idx][1], exp_dir, fold_idx, **kwargs)
        models.append(model)

        print("Evaluating on train")
        train_predictions, train_probabilities, train_scores = evaluate_model(
            model, folds[fold_idx][0], predict_func, exp_dir, all_classes, **kwargs
        )
        print("Evaluating on dev")
        dev_predictions, dev_probabilities, dev_scores = evaluate_model(
            model, folds[fold_idx][1], predict_func, exp_dir, all_classes, **kwargs
        )
        print("Evaluating on test")
        test_predictions, test_probabilities, test_scores = evaluate_model(
            model, folds[fold_idx][2], predict_func, exp_dir, all_classes, **kwargs
        )

        # Save scores to disk
        print("Writing scores to disk")
        with open(
            os.path.join(evaluation_dir, "evaluation.%d.json" % (fold_idx)), "w"
        ) as eval_file:
            results_obj = {
                "train": {
                    "predictions": train_predictions,
                    "probabilities": train_probabilities,
                    "scores": train_scores
                },
                "dev": {
                    "predictions": dev_predictions,
                    "probabilities": dev_probabilities,
                    "scores": dev_scores
                },
                "test": {
                    "predictions": test_predictions,
                    "probabilities": test_probabilities,
                    "scores": test_scores
                },
            }

            json.dump(convert_to_json(results_obj), eval_file)

        all_train_evaluations.append(([label for _, label in folds[fold_idx][0]], train_predictions))
        all_dev_evaluations.append(([label for _, label in folds[fold_idx][1]], dev_predictions))
        all_test_evaluations.append(([label for _, label in folds[fold_idx][2]], test_predictions))

        print(
            "Fold %02d Train: %0.3f (accuracy), %0.3f (precision), %0.3f (recall), %0.3f (f1)"
            % (fold_idx, train_scores['accuracy'], *train_scores['prf_micro'])
        )
        print(
            "Fold %02d Dev: %0.3f (accuracy), %0.3f (precision), %0.3f (recall), %0.3f (f1)"
            % (fold_idx, dev_scores['accuracy'], *dev_scores['prf_micro'])
        )
        print(
            "Fold %02d Test: %0.3f (accuracy), %0.3f (precision), %0.3f (recall), %0.3f (f1)"
            % (fold_idx, test_scores['accuracy'], *test_scores['prf_micro'])
        )

    print()
    print("Overall result aggregated on test")
    print("=================================")
    overall_train_results = compute_aggregate_scores(
        all_train_evaluations, all_classes
    )
    overall_dev_results = compute_aggregate_scores(
        all_dev_evaluations, all_classes
    )
    overall_test_results = compute_aggregate_scores(
        all_test_evaluations, all_classes
    )
    print("Accuracy: %0.3f" % (overall_test_results["accuracy"]))
    print("Micro Precision: %0.3f" % (overall_test_results["prf_micro"][0]))
    print("Micro Recall: %0.3f" % (overall_test_results["prf_micro"][1]))
    print("Micro F1: %0.3f" % (overall_test_results["prf_micro"][2]))
    print("Confusion Matrix")
    print(overall_test_results["confusion_matrix"])

    with open(
        os.path.join(evaluation_dir, "evaluation.overall.json"), "w"
    ) as eval_file:
        json.dump(convert_to_json({
            'train': overall_train_results,
            'dev': overall_dev_results,
            'test': overall_test_results
        }), eval_file)
