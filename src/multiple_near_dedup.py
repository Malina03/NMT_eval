#!/usr/bin/env python

'''Do near-deduplication on parallel sentences based on source, target, both or either. 
    If a single file -main_file- is provided, then do near-deduplication on it. 
    If additional files are provided, then do near-dedup on each file independently, 
    and then remove the sentences that appear in the main file.'''

import sys
import argparse
from unicodedata import category as cat
from unidecode import unidecode
from xxhash import xxh64

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-main_file", "--main_file", required=True, type=str, help="Tab separated input file. This is the main file from which sentences will not be deleted when they appear in other files.")
    parser.add_argument("-other_files", "--other_files", required=False, type=str, nargs='*', help="Tab separated input file. These are the other files from which sentences will be deleted if they appear in the main file.")
    parser.add_argument("-d", "--dedup", default="either", type=str, choices=["src", "source", "tgt", "target", "both", "either"],
                        help="Do we deduplicate based on source, target, both or either")
    args = parser.parse_args()
    return args

def main_near_dedup(file, out_file, remove_non_alpha, dedup):
    # Remove near duplicates, only print to out file the lines that were not duplicates
    hashes = {}
    hashes2 = {}
    with open(out_file, 'w', encoding="utf-8") as out:
        for line in open(file, 'r', encoding="utf-8"):
            if dedup == "either":
                # We deduplicate if either source or target already exists in the corpus
                sent1 = line.strip("\n").lower().split('\t')[0].strip().translate(remove_non_alpha)
                sent2 = line.strip("\n").lower().split('\t')[1].strip().translate(remove_non_alpha)
                tmp1 = xxh64(unidecode(sent1)).hexdigest()
                hsh1 = tmp1.split('\t')[-1]
                tmp2 = xxh64(unidecode(sent2)).hexdigest()
                hsh2 = tmp2.split('\t')[-1]
                if hsh1 not in hashes and hsh2 not in hashes2:
                    # print to out file
                    out.write(line.strip())
                    hashes[hsh1] = 1
                    hashes2[hsh2] = 1
            else:
                # Otherwise we do it on both, source,
                if dedup == "both":
                    sent1 = line.strip("\n").lower().split('\t')[0].strip().translate(remove_non_alpha)
                    sent2 = line.strip("\n").lower().split('\t')[1].strip().translate(remove_non_alpha)
                    sent = sent1 + '\t' + sent2
                elif dedup in ["source", "src"]:
                    sent = line.strip("\n").lower().split('\t')[0].strip().translate(remove_non_alpha)
                elif dedup in ["target", "tgt"]:
                    sent = line.strip("\n").lower().split('\t')[1].strip().translate(remove_non_alpha)
                else:
                    raise ValueError("Illegal argument for -d")
                # Hash and check
                tmp = xxh64(unidecode(sent)).hexdigest()
                hsh = tmp.split('\t')[-1]
                if hsh not in hashes:
                    # print to out file
                    out.write(line.strip())
                    hashes[hsh] = 1
    return hashes, hashes2    

def additional_near_dedup(file, out_file, hashes, hashes2, remove_non_alpha, dedup):
        with open(out_file, 'w', encoding="utf-8") as out:
            for line in open(file, 'r', encoding="utf-8"):
                if dedup == "either":
                    # We deduplicate if either source or target already exists in the corpus
                    sent1 = line.strip("\n").lower().split('\t')[0].strip().translate(remove_non_alpha)
                    sent2 = line.strip("\n").lower().split('\t')[1].strip().translate(remove_non_alpha)
                    tmp1 = xxh64(unidecode(sent1)).hexdigest()
                    hsh1 = tmp1.split('\t')[-1]
                    tmp2 = xxh64(unidecode(sent2)).hexdigest()
                    hsh2 = tmp2.split('\t')[-1]
                    if hsh1 not in hashes and hsh2 not in hashes2:
                        # print to out file
                        out.write(line.strip())
                        hashes[hsh1] = 1
                        hashes2[hsh2] = 1
                else:
                    # Otherwise we do it on both, source,
                    if dedup == "both":
                        sent1 = line.strip("\n").lower().split('\t')[0].strip().translate(remove_non_alpha)
                        sent2 = line.strip("\n").lower().split('\t')[1].strip().translate(remove_non_alpha)
                        sent = sent1 + '\t' + sent2
                    elif dedup in ["source", "src"]:
                        sent = line.strip("\n").lower().split('\t')[0].strip().translate(remove_non_alpha)
                    elif dedup in ["target", "tgt"]:
                        sent = line.strip("\n").lower().split('\t')[1].strip().translate(remove_non_alpha)
                    else:
                        raise ValueError("Illegal argument for -d")
                    # Hash and check
                    tmp = xxh64(unidecode(sent)).hexdigest()
                    hsh = tmp.split('\t')[-1]
                    if hsh not in hashes:
                        # print to out file
                        out.write(line.strip())
                        hashes[hsh] = 1

if __name__ == "__main__":
    args = create_arg_parser()

    # Translate table to remove non alphabetic characters
    tbl = [chr(i) for i in range(sys.maxunicode) if not cat(chr(i)).startswith('L')]
    remove_non_alpha = str.maketrans('', '', ''.join(tbl))

    # Do near-deduplication on main file
    hashes, hashes2 = main_near_dedup(args.main_file, args.main_file + ".dedup", remove_non_alpha, args.dedup)

    # Do near-deduplication on other files, using the hashes from the main file also
    if args.other_files:
        for f in args.other_files:
            additional_near_dedup(f, f + ".dedup", hashes, hashes2, remove_non_alpha, args.dedup)