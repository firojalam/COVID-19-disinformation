
"""
# @author Firoj Alam
# @date July 27, 2020

"""


import csv
import glob
import os
import json
import warnings
import datetime
import argparse
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
from sklearn.metrics import confusion_matrix



class Instance(object):
    def __init__(self, id=1, label="",conf=None):
        self.id = id
        self.class_label = label
        self.conf = conf



def compute_performance(classified_file,gold_file):
    # out_file = open(output_file, "w")
    # out_file.write("class_label\tconfidence\n")
    # index=0

    with open(classified_file) as f:
        classified_data = pd.read_csv(f, sep='\t')

    with open(gold_file) as f:
        gold_file_data = pd.read_csv(f, sep='\t')

    y_true = gold_file_data['class_label'].values.tolist()
    y_pred = classified_data['class_label'].values.tolist()

    acc=metrics.accuracy_score(y_true,y_pred)
    P=metrics.precision_score(y_true,y_pred,average="weighted")
    R=metrics.recall_score(y_true,y_pred,average="weighted")
    F1_w=metrics.f1_score(y_true,y_pred,average="weighted")
    F1_m = metrics.f1_score(y_true, y_pred, average="micro")


    report = metrics.classification_report(y_true, y_pred)
    conf_mat=metrics.confusion_matrix(y_true, y_pred)

    base_name = os.path.basename(classified_file)
    base_name = os.path.splitext(base_name)[0]


    print("{}\t{}\t{}\t{}\t{}\t{}".format(base_name, acc, P, R, F1_w, F1_m))
    print(report)
    print(conf_mat)

    #     out_file.write(final_label+"\t"+str(conf_value)+"\n")
    #
    # out_file.close()




def main():
    warnings.filterwarnings("ignore")
    a = datetime.datetime.now().replace(microsecond=0)

    parser = argparse.ArgumentParser()
    # General Options
    parser.add_argument('-c',"--classified-file", type=str, required=True)
    parser.add_argument('-g', "--gold-file", type=str, required=True)
    # parser.add_argument('-o',"--output-file", type=str, required=True)
    args = parser.parse_args()

    compute_performance(args.classified_file,args.gold_file)

    # all_data_pd.to_csv(args.output_file,sep='\t',index = False, header=True)

    b = datetime.datetime.now().replace(microsecond=0)
    print ("time taken:")
    print(b - a)

if __name__ == '__main__':

    main()


