#!/usr/bin/env python

'''Do near-deduplication on parallel sentences based on source, target, both or either'''

import sys
import argparse
from unicodedata import category as cat
from unidecode import unidecode
from xxhash import xxh64


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_file", required=True, type=str,
                        help="Tab separated input file")
    parser.add_argument("-d", "--dedup", default="either", type=str, choices=["src", "source", "tgt", "target", "both", "either"],
                        help="Do we deduplicate based on source, target, both or either")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_arg_parser()

    # Translate table to remove non alphabetic characters
    tbl = [chr(i) for i in range(sys.maxunicode) if not cat(chr(i)).startswith('L')]
    remove_non_alpha = str.maketrans('', '', ''.join(tbl))

    # Remove near duplicates, only print ones that were not duplicates
    hashes = {}
    hashes2 = {}
    for line in open(args.in_file, 'r', encoding="utf-8"):
        if args.dedup == "either":
            # We deduplicate if either source or target already exists in the corpus
            sent1 = line.strip("\n").lower().split('\t')[0].strip().translate(remove_non_alpha)
            sent2 = line.strip("\n").lower().split('\t')[1].strip().translate(remove_non_alpha)
            tmp1 = xxh64(unidecode(sent1)).hexdigest()
            hsh1 = tmp1.split('\t')[-1]
            tmp2 = xxh64(unidecode(sent2)).hexdigest()
            hsh2 = tmp2.split('\t')[-1]
            if hsh1 not in hashes and hsh2 not in hashes2:
                print(line.strip())
                hashes[hsh1] = 1
                hashes2[hsh2] = 1
        else:
            # Otherwise we do it on both, source,
            if args.dedup == "both":
                sent1 = line.strip("\n").lower().split('\t')[0].strip().translate(remove_non_alpha)
                sent2 = line.strip("\n").lower().split('\t')[1].strip().translate(remove_non_alpha)
                sent = sent1 + '\t' + sent2
            elif args.dedup in ["source", "src"]:
                sent = line.strip("\n").lower().split('\t')[0].strip().translate(remove_non_alpha)
            elif args.dedup in ["target", "tgt"]:
                sent = line.strip("\n").lower().split('\t')[1].strip().translate(remove_non_alpha)
            else:
                raise ValueError("Illegal argument for -d")
            # Hash and check
            tmp = xxh64(unidecode(sent)).hexdigest()
            hsh = tmp.split('\t')[-1]
            if hsh not in hashes:
                print(line.strip())
                hashes[hsh] = 1
