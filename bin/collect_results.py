import argparse
import sys
import json

from pathlib import Path

def main():
    available_metrics = ["accuracy", "micro-precision", "micro-recall", "micro-f1", "macro-precision", "macro-recall", "macro-f1", "weighted-precision", "weighted-recall", "weighted-f1"]
    parser = argparse.ArgumentParser()

    parser.add_argument('--set', type=str, default="test")
    parser.add_argument('--metrics', type=str, default="accuracy, macro-f1, weighted-f1", 
        help="comma separated list of metrics to output. Choices: %s" % (",".join(available_metrics)))
    parser.add_argument('base_exp_directory', type=str)

    args = parser.parse_args()

    # Check arguments
    assert args.set in ['train', 'dev', 'test'], "Invalid set"
    metrics = [m.strip() for m in args.metrics.split(',')]
    for m in metrics:
        if m not in available_metrics:
            print("%s is not a valid metric" % (m))
            print("Available metrics: %s" % (",".join(available_metrics)))
            sys.exit(1)

    base_dir = args.base_exp_directory

    rows = []
    for result_path in Path(base_dir).glob('**/evaluation/evaluation.overall.json'):
        exp_name = "/".join(result_path.parts[:-2])[len(base_dir):]
        with open(result_path) as result_file:
            results = json.load(result_file)
            results = results[args.set]

        row = [exp_name]
        for metric in metrics:
            if metric == "accuracy":
                row.append(results["accuracy"])
            elif metric == "micro-precision":
                row.append(results["prf_micro"][0])
            elif metric == "micro-recall":
                row.append(results["prf_micro"][1])
            elif metric == "micro-f1":
                row.append(results["prf_micro"][2])
            elif metric == "macro-precision":
                row.append(results["prf_macro"][0])
            elif metric == "macro-recall":
                row.append(results["prf_macro"][1])
            elif metric == "macro-f1":
                row.append(results["prf_macro"][2])
            elif metric == "weighted-precision":
                row.append(results["prf_weighted"][0])
            elif metric == "weighted-recall":
                row.append(results["prf_weighted"][1])
            elif metric == "weighted-f1":
                row.append(results["prf_weighted"][2])

        row = list(map(str, row))
        rows.append(row)
    
    # Sort results by experiment name
    rows = sorted(rows, key=lambda r: r[0])
    
    # Print everything
    print("\t".join(['Experiment Name'] + metrics))
    for row in rows:
        print("\t".join(row))

if __name__ == "__main__":
    main()