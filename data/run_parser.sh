#!/bin/bash
export PATH=/nlp/pkg/bin:$PATH

export PYTHONPATH=/nlp/pkg/sw/spacy/lib/python2.7/site-packages:/nlp/pkg/lib64/python2.7/site-packages:/nlp/pkg/lib/python2.7/site-packages:/nlp/pkg/sw/stanford_corenlp_pywrapper/lib/python2.7/site-packages:/nlp/pkg/sw/nltk/lib/python2.7/site-packages:/nlp/pkg/sw/mwparserfromhell/lib64/python2.7/site-packages

export JOB_PATH=/home1/b/bensonc/database_nn/data

date

python $JOB_PATH/qa_wiki_parser.py $JOB_PATH/jeopardy_question/question_answer_pairs_$RUN_NUMBER /scratch-shared/users/bensonc/enwiki-20160204-pages-meta-current.xml $JOB_PATH/jeopardy_results/nn_output_$RUN_NUMBER $JOB_PATH/jeopardy_results/nn_output/question_status_$RUN_NUMBER

date
