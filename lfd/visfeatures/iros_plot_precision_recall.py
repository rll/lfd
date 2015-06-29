#!/usr/bin/env python

import argparse, pickle, os
import matplotlib.pyplot as plt

P_R_FOLDER = 'saved_precision_and_recalls'
#FILE_NAMES = ['original', 'label-0', 'label-1']
#LABELS = ['TPS-RPM (original)', 'TPS-RPM + appearance prior', 'TPS-RPM + appearance prior\n+ label distance']
FILE_NAMES = ['original', 'label-1']
LABELS = ['TPS-RPM (original)', 'TPS-RPM + appearance prior']
#FILE_NAMES = ['score_c1', 'score_c10', 'score_c100', 'score_c01']

def main():
    #parser = argparse.ArgumentParser()
    #args = parser.parse_args()

    plot_setting_set = [('r', '-'), ('c', '-'), ('g', '-'), ('b','-')]

    precision_and_recall = {}

    for fname in FILE_NAMES:
        if os.path.isfile(os.path.join(P_R_FOLDER, fname + '.p')):
            with open(os.path.join(P_R_FOLDER, fname + '.p'), 'rb') as g:
                precision_and_recall[fname] = pickle.load(g)  # (avg_precisions, avg_recalls)

    plt.figure(figsize=(10,5))
    #for index, compare_case in zip(range(len(tps_compare_statistics)), tps_compare_statistics.keys()):
    for (index, fname) in enumerate(FILE_NAMES):
        if fname not in precision_and_recall:
            continue
        (avg_precisions, avg_recalls) = precision_and_recall[fname]
        plt.plot(avg_recalls, avg_precisions, plot_setting_set[index][0], linestyle=plot_setting_set[index][1], label=LABELS[index])
        plt.legend(loc='lower left')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()

    plt.figure(figsize=(10,5))
    #for index, compare_case in zip(range(len(tps_compare_statistics)), tps_compare_statistics.keys()):
    for (index, fname) in enumerate(FILE_NAMES):
        if fname not in precision_and_recall:
            continue
        (avg_precisions, avg_recalls) = precision_and_recall[fname]
        plt.plot(range(1,len(avg_precisions)+1), avg_precisions, plot_setting_set[index][0], linestyle=plot_setting_set[index][1], label=fname+' TPS-RPM')
        plt.legend(loc='upper right')
    plt.xlabel('retrieval number')
    plt.ylabel('precision')
    plt.show()

if __name__ == "__main__":
    main()
