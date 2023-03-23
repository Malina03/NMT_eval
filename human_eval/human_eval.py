#!/usr/bin/env python

'''Analyse human evaluations of NMT systems
   Expects tab-separated input files of the following format:
   source doc_id mt1-without-macocu mt2-with-macocu src_lang tgt_lang eval desc eval_details time'''

import argparse
import ast
import copy
from sklearn.metrics import cohen_kappa_score
from scipy.stats import binomtest


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_files", nargs="+", required=True, type=str,
                        help="Input files with annotations")
    args = parser.parse_args()
    return args


def load_data(in_f):
    '''Load tsv data of keops output of annotators
       Files are tab-separated, with this header:
       source doc_id mt1-without-macocu mt2-with-macocu src_lang tgt_lang eval desc eval_details time
       "desc" and "eval_details" are empty'''
    data = []
    times = []
    for idx, line in enumerate(open(in_f, 'r', encoding="utf-8")):
        if idx > 0:
            tab = line.strip().split('\t')
            times.append(int(tab[9]))
            # We only care about evaluation (literal python dict) and time, for now
            d = ast.literal_eval(tab[6].strip())
            if int(d["mt1-without-macocu"]) > int(d["mt2-with-macocu"]):
                anno = "with_mcc"
            elif int(d["mt1-without-macocu"]) == int(d["mt2-with-macocu"]):
                anno = "tie"
            elif int(d["mt1-without-macocu"]) < int(d["mt2-with-macocu"]):
                anno = "without_mcc"
            data.append(anno)
    return data, times


def print_general_stats(annos, times, in_f):
    '''Print general statistics for this data (e.g. how often with > without)'''
    mcc_with = annos.count("with_mcc")
    mcc_without = annos.count("without_mcc")
    ties = annos.count("tie")
    print (f"For {in_f}:\n")
    with_pct = round((float(mcc_with) / float(len(annos))) * 100, 1)
    without_pct = round((float(mcc_without) / float(len(annos))) * 100, 1)
    tie_pct = round((float(ties) / float(len(annos))) * 100, 1)
    avg_seconds = round(float(sum(times)) / float(len(times)), 1)
    print (f"Annotator(s) took {avg_seconds} seconds on average")
    print (f"with > without: {mcc_with} ({with_pct}%)")
    print (f"with == without: {ties} ({tie_pct}%)")
    print (f"with < without: {mcc_without} ({without_pct}%)")

    # Do a binomial test that with is better than without
    # We have ties, so add half of them to the successrate
    res = binomtest(mcc_with + int((0.5 * ties)), n=len(annos), p=0.5, alternative="two-sided")
    print (f"p-value of binomtest: {round(res.pvalue, 3)}\n")


def annotator_agreement(anno1, anno2, in_f1, in_f2):
    '''Calculate cohens kappa inter-annotator agreement'''
    score = round(cohen_kappa_score(anno1, anno2), 2)
    agree = sum([1 for idx, val in enumerate(anno1) if val == anno2[idx]])
    agree_pct = round((float(agree) / float(len(anno1)) * 100), 1)
    # Calculate number of times where 1 thought m1 > m2 and the other m2 < m1
    hard_disagree = 0
    for a1, a2 in zip(anno1, anno2):
        if (a1 == "without_mcc" and a2 == "with_mcc") or a2 == "without_mcc" and a1 == "with_mcc":
            hard_disagree += 1
    hard_disagree_pct = round((float(hard_disagree) / float(len(anno1)) * 100), 1)

    print (f"For annotators {in_f1} and {in_f2}:\n")
    print (f"Agreement: {agree} times ({agree_pct}%)")
    print (f"Hard disagree: {hard_disagree} times ({hard_disagree_pct}%)")
    print (f"Cohen Kappa score: {score}\n")


def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()
    all_annotations = []
    all_times = []
    # Loop over input files and get the data
    for in_f in args.input_files:
        lang = in_f[0:2]
        annotations, times = load_data(in_f)
        all_annotations.append(annotations)
        all_times.append(times)
        # Print general statistics for this data (e.g. how often with > without)
        print_general_stats(annotations, times, in_f)

    # Print statistics for the files together
    # E.g. see how we do on a full language for all annotators
    full = copy.deepcopy(all_annotations[0])
    full_times = copy.deepcopy(all_times[0])
    for idx in range(1, len(all_annotations)):
        full += all_annotations[idx]
        full_times += all_times[idx]
    print_general_stats(full, full_times, lang + " total")

    # If we have two or three files, do inter annotator agreement with cohens kappa
    if len(args.input_files) == 2:
        annotator_agreement(all_annotations[0], all_annotations[1], args.input_files[0], args.input_files[1])
    elif len(args.input_files) == 3:
        annotator_agreement(all_annotations[0], all_annotations[1], args.input_files[0], args.input_files[1])
        annotator_agreement(all_annotations[0], all_annotations[2], args.input_files[0], args.input_files[2])
        annotator_agreement(all_annotations[1], all_annotations[2], args.input_files[1], args.input_files[2])


if __name__ == '__main__':
    main()
