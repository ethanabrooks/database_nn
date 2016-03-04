from __future__ import print_function

import xml.sax
import mwparserfromhell
import sys
import io
from spacy.en import English
from wiki_nlp_util import *


# qa_object is an object that encompasses the question/answer pair
class qa_object:
    def __init__(self, question, answer, entity):
        self.question = question
        self.answer = answer
        self.entity = entity
        self.correct_sents = []
        self.wrong_sents = []

    def store_correct_sents(self, sents):
        self.correct_sents = sents

    def store_wrong_sents(self, sents):
        self.wrong_sents = sents

    # Testing purposes
    def print(self):
        print(
            'Question: ' + self.question +
            ' \n Answer: ' + self.answer +
            ' \n Entity: ' + self.entity
        )

    # no line breaks
    def get_clean_sentence(self, s):
        return s.replace('\n', ' ').replace('\r', '')

    def get_qa_string(self):
        if len(self.correct_sents) == 0 or len(self.wrong_sents) == 0:
            return ''
        output = self.question
        output += '\n' + self.answer
        # Only print first correct sentence
        for s in self.correct_sents:
            output += '\n' + self.get_clean_sentence(s)
            break
        output += '\n' + str(len(self.wrong_sents))
        index = 1
        for s in self.wrong_sents:
            output += '\n' + self.get_clean_sentence(s)
            index += 1
        return output


# Takes in a file where every line contains the question and answer separated by a '|'
# Returns a dictionary where the key is the entity extracted, and the value is the qa_object
# nlp_parser is the spacy nlp parser object
def read_question_answer_pair(filename, nlp_parser):
    with open(filename) as f:
        lines = f.read().splitlines()
    d = {}
    for line in lines:
        decoded_line = line.decode('utf-8').strip()
        l = decoded_line.split('|')
        entities = get_ents_from_string(nlp_parser(l[0]))
        for entity in entities:
            # does not account for the fact that the same entity can appear in different questions!
            # Also entities are indexed by their lowercase forms (not case sensitive)
            entity_name = entity.text
            d[entity_name.lower()] = qa_object(l[0], l[1], entity_name)
    return d


# Class that extends ContentHandler
# Ignores cases when comparing to document
class WikiContentHandler(xml.sax.ContentHandler):
    def __init__(self, nlp_parser, entity_pairs):
        xml.sax.ContentHandler.__init__(self)
        self.nlp_parser = nlp_parser
        # entity_answer pairs from read_entity_answer_pair function
        self.entity_pairs = entity_pairs

        # Page check
        self.page_flag = False
        # Title check
        self.title_flag = False
        # Text check
        self.text_flag = False
        # Entity matched
        self.entity_flag = False

        # Keep track of the current matched entity (title) and text
        self.current_entity = ''
        self.current_entity_text = ''

    # Removes wikipedia markup using external library
    def clean_markup(self, content):
        clean_code = mwparserfromhell.parse(content).strip_code()
        return clean_code

    # Updates the entity/object pair by storing the correct/wrong sentence results
    def update_entity_from_wiki_text(self):
        wiki_text = self.clean_markup(self.current_entity_text)
        parsed_wiki_text = self.nlp_parser(wiki_text)

        qa_object = self.entity_pairs[self.current_entity]

        correct_sents = filter_sentences(parsed_wiki_text, qa_object.answer)
        incorrect_sents = []
        if correct_sents != []:
            incorrect_sents = fetch_random_sentences(parsed_wiki_text, qa_object.answer, 15)
        qa_object.store_correct_sents(correct_sents)
        qa_object.store_wrong_sents(incorrect_sents)

        self.entity_pairs[self.current_entity] = qa_object
        return

    def startElement(self, name, attrs):
        if name == 'page':
            self.page_flag = True
        elif name == 'title':
            self.title_flag = True
        elif name == 'text':
            self.text_flag = True

    def endElement(self, name):
        if name == 'page':
            self.page_flag = False

            # If entity has been matched, reset state
            if self.entity_flag:
                self.update_entity_from_wiki_text()
                self.entity_flag = False
                self.current_entity = ''
                self.current_entity_text = ''
        elif name == 'title':
            self.title_flag = False
        elif name == 'text':
            self.text_flag = False

    def characters(self, content):
        if self.title_flag:
            title = content.lower()
            # the keys in entity_pairs are already lowercase
            if title in self.entity_pairs:
                self.entity_flag = True
                self.current_entity = title
        elif self.text_flag:
            if self.entity_flag:
                self.current_entity_text += content


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Not enough arguments given', file=sys.stderr)
        sys.exit()

    input_filename = sys.argv[1]
    xml_sourcename = sys.argv[2]
    output_filename = sys.argv[3]

    nlp_parser = English()
    d = read_question_answer_pair(input_filename, nlp_parser)

    source = open(xml_sourcename)
    handler = WikiContentHandler(nlp_parser, d)
    xml.sax.parse(source, handler)

    pairs = handler.entity_pairs

    output = io.open(output_filename, 'w', encoding='utf-8')

    for entity in pairs:
        entity_string = pairs[entity].get_qa_string()
        if entity_string != '':
            output.write(pairs[entity].get_qa_string() + '\n')
    print("all done")
    source.close()
