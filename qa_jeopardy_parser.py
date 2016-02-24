import codecs
import json

import sys

import re

from numpy.ma import sqrt


def delete_markup(text):
    global mean
    global n
    global cumulative_delta
    pattern = re.compile(r'<[^>]*>')
    sub = pattern.sub('', text)

    n += 1.0
    delta = len(sub) - mean
    mean += delta/n
    cumulative_delta += delta*(len(sub) - mean)
    return sub


def filter_for_qa(json_obj):
    return {key: delete_markup(json_obj[key])
            for key in json_obj
            if key in ('category', 'question', 'answer')}


if __name__ == '__main__':
    n = 0.0
    mean = 0.0
    cumulative_delta = 0.0
    with codecs.open(sys.argv[1], encoding='utf-8') as json_file:
        qa_list = json.load(json_file, object_hook=filter_for_qa)
        with codecs.open(sys.argv[2], encoding='utf-8', mode='w+') as qa_file:
            for qa in qa_list:
                qa_string = (qa['category'] + ": " +
                             qa['question'].strip("'") + "|" +
                             qa['answer'] + "\n").replace("\\", "")
                qa_file.write(qa_string)
    stdev = sqrt(cumulative_delta / (n-1))
    print(n)
    print(mean)
    print(stdev)
