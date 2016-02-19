from spacy.en import English
import random

#Given the nlp_parser and a string, returns a list of named entities in the string
def get_ents_from_string(parsed_string):
 	return list(parsed_string.ents)

# Filter sentences to match key word, returns valid sentences
def filter_sentences(doc, word):
	key_word_lower = word.lower()
	valid_sents = []
	for s in doc.sents:
		if key_word_lower in s.text.lower():
			valid_sents.append(s.text.rstrip())
	return valid_sents

# Fetch random sentences without keyword
def fetch_random_sentences(doc, omit_word, num_to_retrieve):
	invalid_sents = []
	key_word_lower = omit_word.lower()
	for s in doc.sents:
		if (not (key_word_lower in s.text.lower())) and (random.uniform(0, 1) > 0.8):
			invalid_sents.append(s.text.rstrip())
		if len(invalid_sents) >= num_to_retrieve:
			break
	return invalid_sents
