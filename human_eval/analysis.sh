#!/bin/bash

# Loop over files of a certain language and print the analysis
# Works because we prefixed the annotation file with the language
for lang in bg hr is mt sl tr ; do 
	files=$(ls ${lang}*)
	python human_eval.py --input_files $files
done
