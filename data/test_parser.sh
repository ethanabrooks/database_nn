#!/bin/bash
export PATH=/nlp/pkg/bin:$PATH

export PYTHONPATH=/nlp/pkg/sw/spacy/lib/python2.7/site-packages:/nlp/pkg/lib64/python2.7/site-packages:/nlp/pkg/lib/python2.7/site-packages:/nlp/pkg/sw/stanford_corenlp_pywrapper/lib/python2.7/site-packages:/nlp/pkg/sw/nltk/lib/python2.7/site-packages:/nlp/pkg/sw/mwparserfromhell/lib64/python2.7/site-packages

which python

export JOB_PATH=/home1/b/bensonc/database_nn

python $JOB_PATH/test.py
