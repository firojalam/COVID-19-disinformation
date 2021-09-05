import argparse
import os

import covid_exp_utils
import fasttext

############################# INTERFACE DEFINITION #############################
def custom_train_func(
    train_fold,
    dev_fold,
    exp_dir,
    fold_idx,
    custom_arg_1=None,
    custom_arg_2=None,
    custom_arg_3=None,
    **kwargs
):
    # use parameters to initialize, train, save and return a model
    #
    # Arguments:
    # ==========
    # train_fold: list of (instance, label) tuples
    # dev_fold: list of (instance, label) tuples
    # exp_dir: Experiment directory path (see example below of using this path to
    #            save models in exp_dir/models)
    # fold_idx: Fold index
    #
    # Returns:
    # ========
    # model: trained model
    #
    # Notes:
    # ======
    # First line of arguments is hardcoded (train_fold, dev ..., fold_idx)
    # Second line of arguments are custom arguments your model might need
    #    (see example below)
    # Third line (**kwargs) is to collect all remaining arguments (leave as is)
    pass


def custom_predict_func(
    model,
    instances,
    exp_dir,
    custom_arg_1=None,
    custom_arg_2=None,
    custom_arg_3=None,
    **kwargs
):
    # use parameters to pass instances through your model and output predictions
    #    and probabilities
    #
    # Arguments:
    # ==========
    # model: trained model
    # instances: list of instances (strings)
    # exp_dir: Experiment directory path
    #
    # Returns:
    # ========
    # predictions: list of strings (one element per instance)
    # probabilities: list of floats (one element per instance)
    #
    # First line of arguments is hardcoded (model, instances, exp_dir)
    # Second line of arguments are custom arguments your model might need
    #    (example below takes no custom arguments, which is also fine)
    # Third line (**kwargs) is to collect all remaining arguments (leave as is)
    pass


################################################################################


def fasttext_train_func(
    train_fold,
    dev_fold,
    exp_dir,
    fold_idx,
    lr=0.1,
    dim=100,
    ws=5,
    epoch=5,
    wordNgrams=1,
    verbose=2,
    pretrainedVectors="",
    **kwargs
):
    exp_dir, tmp_dir, models_dir, _ = covid_exp_utils.get_experiment_dirs(exp_dir)

    # Prepare train and dev in fasttext format
    train_file = os.path.join(tmp_dir, "train.txt")
    with open(train_file, "w") as processed_file:
        for instance, label in train_fold:
            processed_file.write("__label__%s %s\n" % (label, instance))

    dev_file = os.path.join(tmp_dir, "dev.txt")
    with open(dev_file, "w") as processed_file:
        for instance, label in dev_fold:
            processed_file.write("__label__%s %s\n" % (label, instance))

    model = fasttext.train_supervised(
        input=train_file,
        lr=lr,
        dim=dim,
        ws=ws,
        epoch=epoch,
        wordNgrams=wordNgrams,
        verbose=verbose,
        pretrainedVectors=pretrainedVectors,
    )

    # Save model to disk
    model.save_model(os.path.join(models_dir, "model.%d.bin" % (fold_idx)))

    return model


def fasttext_predict_func(model, instances, exp_dir, **kwargs):
    predictions, probabilities = model.predict(instances)
    predictions = [pred[0][len(model.label) :] for pred in predictions]
    probabilities = [prob[0] for prob in probabilities]

    return predictions, probabilities


def main():
    parser = argparse.ArgumentParser()

    # General Options
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--exp-dir", type=str, required=True)

    # Fasttext specific options
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--dim", default=100, type=int)
    parser.add_argument("--ws", default=5, type=int)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--wordNgrams", default=1, type=int)
    parser.add_argument("--verbose", default=2, type=int)
    parser.add_argument("--pretrainedVectors", default="", type=str)

    args = parser.parse_args()

    covid_exp_utils.prepare_experiment_dirs(args.exp_dir)

    folds = covid_exp_utils.load_data_splits(args.data_dir)

    covid_exp_utils.run_model(
        args.exp_dir,
        folds,
        fasttext_train_func,
        fasttext_predict_func,
        lr=args.lr,
        dim=args.dim,
        ws=args.ws,
        epoch=args.epoch,
        wordNgrams=args.wordNgrams,
        verbose=args.verbose,
        pretrainedVectors=args.pretrainedVectors,
    )


if __name__ == "__main__":
    main()
