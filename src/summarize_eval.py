#!/usr/bin/env python
# -*- coding: utf8 -*-

'''Summarize all evaluation metrics (that were printed to a file) and print in a nice table

   Input folder can either be a folder with experiments, e.g. data/exp/, in which we expect e.g.:

   data/exp/exp1/output/
   data/exp/exp2/output/
   data/exp/exp3/output/
   ...
   To exist. But it you can also specify data/exp1/ directly, for just a single overview'''

import sys
import argparse
import os
import re
from tabulate import tabulate


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", required=True, type=str,
                        help="Main input folder")
    # Options here: https://pypi.org/project/tabulate/
    parser.add_argument("-t", "--tablefmt", default="fancy_grid",
                        help="Output format of the table, default 'fancy_grid. \
                              Useful options: latex, latex_booktabs, tsv, pretty, github")
    # Experiment files we check
    parser.add_argument("-e", "--experiments", nargs="+", default=["dev_ep", "flores_devtest.out", "flores_dcmt", "flores_devtest_dcmt.out", "wmt16", "wmt17", "wmt18", "wmt_dev", "wmt_test", "TED", "Wiki", "QED"],
                        help="Experiments to print scores for: default is all of them, provided they have a single score")
    # Only print specified metrics, default we print all of them
    parser.add_argument("-m", "--metrics", nargs="+", default=["bleu", "ter", "chrf", "chrfpp", "comet", "bleurt", "bertscore"],
                        help="Metrics to print scores for - default is all of them")
    # For reproducibility it can be nice to print the call back
    parser.add_argument("-pc", "--print_call", action="store_true", help="Print python call back")
    args = parser.parse_args()
    return args


def select_eval_file(met, eval_files, contains):
    '''Select the correct eval file based on the metric'''
    evals = [x for x in eval_files if x.endswith(met) and x.split('/')[-1].lower().startswith(contains.lower())]
    if not evals:
        #print(f"WARNING: file for metric {met} not found - return score of 0")
        return []
    if len(evals) == 1:
        return evals[0]

    # More files are found, only take the one with the highest epoch
    print(f"WARNING: more eval files found for metric {met}, select highest epoch")
    cur_ep = -1
    highest_file = ''
    for eval_file in evals:
        # Bit hacky, select last number found as epoch
        epoch = int(re.findall(r'\d+', eval_file)[-1])
        if epoch > cur_ep:
            highest_file = eval_file
            cur_ep = epoch
    return highest_file


def direct_subdirectories(a_dir):
    '''Return direct subdirs for a given dir'''
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


def average_list(in_list):
    '''Average list of numbers'''
    return float(sum(in_list)) / float(len(in_list))


def load_score(eval_file, met):
    '''Load score from an evaluation file and return as a float'''
    scores = [x.strip() for x in open(eval_file, 'r')]
    # Score is just the only value in the file
    if met in ['bleu', 'ter', 'chrf', 'chrfpp']:
        score = scores[0]
    elif met in ['comet', 'bertscore']:
        # Score is last value of last line for comet
        # For bertscore, the file looks like this, only extract F1 for now
        # xlm-roberta-large_L17_no-idf_version=0.3.11(hug_trans=4.4.2)-rescaled_fast-tokenizer \
        # P: 0.934309 R: 0.930420 F1: 0.932313
        # We multiply by 100 so all score similarly interpretable
        score = float(scores[-1].split()[-1]) * 100
    elif met == 'bleurt':
        # We have to average the scores
        score = average_list([float(x) for x in scores]) * 100
    else:
        raise ValueError(f"Metric {met} is not one of the metrics we know of")
    return float(score)


def get_exp_fols(input_fol):
    '''Get the experiment folders based on the input folder'''
    sub_dirs = direct_subdirectories(input_fol)
    # Check if the subdirs are of a single experiment, or that we look at multiple experiments
    if 'output' in sub_dirs:
        exp_names = [input_fol.split('/')[-2]]
        exp_folders = [input_fol + '/output/']
    else:
        exp_names = sub_dirs
        exp_folders = [os.path.join(input_fol, sub) + '/output/' for sub in sub_dirs]
    return exp_names, exp_folders


def get_scores(exp_names, exp_folders, metrics, out_name, scores, included_exps):
    '''For each experiment, read all eval files and get the final score'''
    include = False
    for name, fol in zip(exp_names, exp_folders):
        if name not in scores:
            scores[name] = []
        # Folder exists (not always certain), so extract eval files
        if os.path.isdir(fol):
            eval_files = [os.path.join(fol, f) for f in os.listdir(fol)
                          if os.path.isfile(os.path.join(fol, f)) and 'eval' in f]
            # First check if we at least find a single eval file
            found = False
            for met in metrics:
                eval_file = select_eval_file(met, eval_files, out_name)
                if eval_file:
                    found = True

            # Loop over metrics and select eval file for the current metric
            if found:
                for met in metrics:
                    eval_file = select_eval_file(met, eval_files, out_name)
                    # Load the actual score here
                    score = load_score(eval_file, met) if eval_file else 0
                    # Get rid of weird float error that transforms 0.55 to 0.549999999999
                    score = round(score, 8)
                    score = round(score, 1)
                    scores[name].append(score)
                    include = True
    # Check if we add this experiment to the ones for which we found results
    if include:
        included_exps.append(out_name)
    return scores, included_exps


def print_table(scores, exp_names, metrics, tablefmt, experiments):
    '''Print a nice table using the score dictionary'''
    print_list = []
    # Read scores per experiment and save in a list
    for name in exp_names:
        if scores[name]:
            print_list.append([name] + scores[name])
    # Sort experiments by first specified metric (default bleu)
    print_list = sorted(print_list, key=lambda x: x[1], reverse=True)
    # Create two headers
    exps = ['Exp']
    headers = ['']
    for idx in range(len(experiments)):
        headers += metrics
        for idx2, met in enumerate(metrics):
            # Only add the experiment name for the first metric
            if idx2 == 0:
                exps.append(experiments[idx].replace('.out', ''))
            else:
                exps.append('')
    print_list.insert(0, headers)
    # Print a nice list with tabulate
    print(tabulate(print_list, headers=exps, tablefmt=tablefmt))


def main():
    '''Main function for summarizing the created evaluation files'''
    args = create_arg_parser()
    if args.print_call:
        print("Produced by:\npython {0}\n".format(" ".join(sys.argv)))

    # Get folders and names based on the main input folder
    exp_names, exp_folders = get_exp_fols(args.input_folder)

    # For each experiment, get the output files and save the scores
    scores = {}
    included_exps = []
    for exp in args.experiments:
        scores, included_exps = get_scores(exp_names, exp_folders, args.metrics, exp, scores, included_exps)

    # We filled a dictionary with scores, print a nice (sorted) table
    print_table(scores, exp_names, args.metrics, args.tablefmt, included_exps)


if __name__ == '__main__':
    main()
