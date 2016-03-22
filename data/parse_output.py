import sys
import io
import os
import random as rand
from spacy.en import English
from wiki_nlp_util import *

DATA_PATH_TRAIN = 'qa_files/training/'
DATA_PATH_TEST = 'qa_files/test/'
DATA_PATH_VALIDATION = 'qa_files/validation/'

class qa_object:
	def __init__(self, question, answer, correct_sent, incorrect_sents):
		self.question = question
		self.answer = answer
		self.correct_sent = correct_sent
		self.incorrect_sents = incorrect_sents

#returns a list of qa_objects
def parse_qa_file(filename):
	with open(filename) as f:
		lines = f.read().splitlines()
	d = []
	i = 0
	while i < len(lines):
		question = lines[i]
		i+=1
		answer = lines[i]
		i+=1
		correct_sent = lines[i]
		i+=1		
		incorrect_sents_num = int(lines[i])
		i+=1
		incorrect_sents = []
		for j in range(incorrect_sents_num):
			incorrect_sents.append(lines[i])
			i+=1
		d.append(qa_object(question, answer, correct_sent, incorrect_sents))
	return d

#Takes in the list of qa_objects returned by parse_qa_file
def parse_into_format(nlp_parser, d, vocab_filename):
	index = 1
	final_vocab_list = set()
	for qa_object in d:
		# tokenized_question = get_tokenized_string(nlp_parser, qa_object.question)
		# tokenized_answer = get_tokenized_string(nlp_parser, qa_object.answer)
		# tokenized_correct_sent = get_tokenized_string(nlp_parser, qa_object.correct_sent)
		# tokenized_incorrect_sents = []

		# for incorr_sent in qa_object.incorrect_sents:
		# 	tokenized_incorrect_sents.append(get_tokenized_string(nlp_parser, incorr_sent))
		data_path = get_data_path(0.60, 0.20, 0.20)
		#Index for individual correct/incorrect sentences in a question/answer pair
		example_index = 0

		# Correct Sentence
		data_path_example = data_path + str(index) + '_' + str(example_index)
		data_file = open(data_path_example, 'w+')
		(write_sucess, vocab_list) = write_example_to_file(data_file, example_index, qa_object.question, qa_object.answer, qa_object.correct_sent, nlp_parser, 1, replace_ents=1)
		data_file.close()

		if (write_sucess == False):
			os.remove(data_path_example)
			continue

		example_index += 1

		#Incorrect Sentences
		random_index = rand.randint(0, len(qa_object.incorrect_sents)-1)
		data_path_example = data_path + str(index) + '_' + str(example_index)
		data_file = open(data_path_example, 'w+')
		(write_sucess, vocab_list_2) = write_example_to_file(data_file, example_index, qa_object.question, qa_object.answer, qa_object.incorrect_sents[random_index], nlp_parser, 0, replace_ents=1)
		data_file.close()

		final_vocab_list = final_vocab_list.union(vocab_list, vocab_list_2)
		#Update question index
		index += 1

	vocab_file = open(vocab_filename, 'w+')
	for vocab in final_vocab_list:
		vocab_file.write(vocab + '\n')

#returns true if sucessfully written, or returns false
def write_example_to_file(data_file, index, question, answer, sentence, nlp_parser, correct, replace_ents=0):
	parsed_sent = nlp_parser(sentence)
	parsed_question = nlp_parser(question)
	parsed_answer = nlp_parser(answer)

	vocab = set()
	if (replace_ents):
		parsed_doc = [parsed_sent, parsed_question, parsed_answer]

		ents_dict = {}

		for doc in parsed_doc:
			for ent in doc.ents:
				if ent.text not in ents_dict:
					ents_dict[ent.text] = ent

		ents_id_tuple = []

		index = 1
		for (ent_text, ent) in ents_dict.items():
			# Make sure the ent is non-empty
			if len(ent_text) > 0:
				ents_id_tuple.append((ent, "@ent" + str(index)))
				index+=1

		(sentence, sent_vocab) = change_ent_in_doc(parsed_sent, ents_id_tuple)
		(question, question_vocab) = change_ent_in_doc(parsed_question, ents_id_tuple)
		(answer, answer_vocab) = change_ent_in_doc(parsed_answer, ents_id_tuple)
		vocab = vocab.union(sent_vocab, question_vocab, answer_vocab)
	else:
		question = get_tokenized_string(nlp_parser, question)
		answer = get_tokenized_string(nlp_parser, answer)
		sentence = get_tokenized_string(nlp_parser, sentence)

	if correct and ((answer not in sentence) or ('@ent' not in answer)):
		return (False, None)

	data_file.write(str(index) + '\n' + '\n')
	data_file.write(sentence + '\n' + '\n')
	data_file.write(question + '\n' + '\n')

	# If incorrect, answer is default the null entity
	if (correct):
		data_file.write(answer + '\n\n')
	else:
		data_file.write('@ent0' + '\n\n')

	if (replace_ents):
		write_ents_map(data_file, ents_id_tuple)
	return (True, vocab)

def write_ents_map(data_file, ents_id_tuple):
	#Append the no answer option
	data_file.write('@ent0:\n')
	for (k, v) in ents_id_tuple:
		data_file.write(v+ ':' + k.text + '\n')

# Also tokenizes the sentences
def change_ent_in_doc(doc, ents_id_tuple):
	s = ''
	#IOB state
	temp_tokens_for_ent = []
	vocab = set()
	for t in doc:
		if (t.ent_iob == 3):
			# print('3: token is: ' + t.lower_)
			s = append_ent_to_string(s, temp_tokens_for_ent, ents_id_tuple)
			temp_tokens_for_ent = []
			temp_tokens_for_ent.append(t)
		elif (t.ent_iob == 1):
			# print('1: token is: ' + t.lower_)
			temp_tokens_for_ent.append(t)
		else:
			# print('2: token is: ' + t.lower_)
			s = append_ent_to_string(s, temp_tokens_for_ent, ents_id_tuple)
			temp_tokens_for_ent = []
			s += t.lower_ + ' '
			vocab.add(t.lower_)
		# Append the ent if there is still one at the end
	s = append_ent_to_string(s, temp_tokens_for_ent, ents_id_tuple)
	return (s[:-1], vocab)

def append_ent_to_string(s, temp_tokens_for_ent, ents_id_tuple):
	if (len(temp_tokens_for_ent) == 0):
		return s
	else:
		for (k, v) in ents_id_tuple:
			if len(temp_tokens_for_ent) != len(k):
				continue
			for i in range(len(k)):
				is_correct = True
				if (k[i].text != temp_tokens_for_ent[i].text):
					is_correct = False
				# Add the entity id
			if is_correct:
				# print('s is: ' + s + ' v added is: ' + v)
				s += v + ' '
				return s

def get_data_path(p_train, p_test, p_val):
	r = rand.random()
	if r < p_train:
		return DATA_PATH_TRAIN
	elif r < p_train + p_test:
		return DATA_PATH_TEST
	else:
		return DATA_PATH_VALIDATION

def get_tokenized_string(nlp_parser, s):
	tokens = nlp_parser(s)
	s = ''
	for t in tokens:
		s += t.lower_ + ' '
	# remove trailing whitespace
	if len(s) > 1:
		s = s[:-1]
	return s

if __name__ == '__main__':
	nlp_parser = English()
	d = parse_qa_file('nn_output')

	test_obj = d[0]
	parse_into_format(nlp_parser, d, 'vocab.txt')
	# for i in range(10):
		# print(d[i].question)
		# print(d[i].answer)
		# print(d[i].correct_sent)
		# print(len(d[i].incorrect_sents))
		# print('done: ' + str(i))