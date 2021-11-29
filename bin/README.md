# COVID-19-disinformation

## Setup Environment
Please create an experimental directory
```bash
mkdir exp_covid19_disinfo
```

To better organize things please copy the data and scripts into *exp_covid19_disinfo* directory.

```bash
cd /your_path/exp_covid19_disinfo/
conda env create -f bin/covid_exp_env.yaml
conda activate covid
```

## Dataset
Please check the Readme in directory data/

## How to Run Experiments

### Experiments with transformers

```bash
cd /your_path/exp_covid19_disinfo/
```

- All transformers scripts are here
```bash
cd /your_path/exp_covid19_disinfo/
export HOME_DIR="/your_path/exp_covid19_disinfo/bin/transformers"
bash bin/run_en_bert.sh
```


### Experiments with FastText
First, download the pre-trained embeddings released by the FastText team.

- **Arabic:** [Common Crawl and Wikipedia CBOW](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ar.300.vec.gz)

- **Bulgarian:** [Common Crawl and Wikipedia CBOW](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bg.300.vec.gz)

- **Dutch:** [Common Crawl and Wikipedia CBOW](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nl.300.vec.gz)

- **English:** [2 million word vectors trained with subword information on Common Crawl (600B tokens).](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip)

After downloading and extracting the embeddings, two `*.vec` files will be available to be used in our experiments.

Run the following command to start an experiment for a specific question:

```bash
bash bin/run_en_fasttext.sh
```

- `pretrained-vectors/crawl-300d-2M-subword.vec` points to the pretrained vector (replace with language specific vectors when running experiments)
- `--autotuneDuration` defines how long the tuning should run in seconds - the longer this is, the better the final model.

### Experiments with Social features

First, install the following packages:

- pip install requests
- pip install feature-engine

Then, go through the notebook `twitter-v2.ipynb`. The notebook reads the social features in `data/english/covid19_infodemic_english_data_multiclass_final_all.jsonl`, and converts them to machine learning format. This means that categorical features are converted via one-hot-encoder technique, numerical features are log scaled, and boolean features are turned to 0s and 1s. The last cell of the notebook saves the output under `data/` folder with a file named `feature_english.tsv`.



### Script to get the results from the generated json files

Run the following script to collect all results within a base experimental directory:

```bash
python bin/collect_results.py --set test --metrics "accuracy, micro-f1, weighted-f1" experiments/exp_bert_arabic/
```

The script supports nested experimental setups, it will automatically find all experiments within this directory for all the questions and output one row per experiment.
