import codecs
import json

import sys


def filter_for_qa(json_obj):
    return {key: json_obj[key] for key in json_obj
            if key in ('question', 'answer')}


if __name__ == '__main__':
    with codecs.open(sys.argv[1], encoding='utf-8') as json_file:
        qa_list = json.load(json_file, object_hook=filter_for_qa)
        with codecs.open(sys.argv[2], encoding='utf-8', mode='w+') as qa_file:
            for qa in qa_list:
                qa_string = (qa['question'].strip("'") + "|" +
                             qa['answer'] + "\n").replace("\\", "")
                qa_file.write(qa_string)
