#!/bin/bash
export PATH=/nlp/pkgs/bin:$PATH

export PYTHONPATH=/nlp/pkg/sw/spacy/lib/python2.7/site-packages:/nlp/pkg/lib64/python2.7/site-packages:/nlp/pkg/lib/python2.7/site-packages:/nlp/pkg/sw/stanford_corenlp_pywrapper/lib/python2.7/site-packages:/nlp/pkg/sw/nltk/lib/python2.7/site-packages:/nlp/pkg/sw/mwparserfromhell/lib64/python2.7/site-packages

python qa_wiki_parser.py question_answer_test_pairs simplewiki-20160203-pages-meta-current.xml output_1
