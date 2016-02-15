from spacy.en import English

# def get_ents_from_string(doc):
# 	tokens = nlp(doc)
# 	return tokens.ents

def fetch_content_from_doc(filename):
	f = open(filename, 'r')
	s = f.read()
	f.close()
	return s

# Filter sentences to match key word,
def filter_sentences(doc, word):
	key_word_lower = word.lower()
	valid_sents = []
	for s in doc.sents:
		for t in s:
			if t.lower_ == key_word_lower:
				valid_sents.append(s.text)
				break
	return valid_sents

def fetch_random_sentences(doc, omit_word, num_to_retrieve):
	n = get_num_sentences(doc)
	invalid_sents = []
	return


# def read_entity_answer_wiki_tuple(filename):
# 	with open(filename) as f:
# 		lines = f.read().splitlines()
# 	d = []
# 	for line in lines:
# 		l = line.split(',', 2)
# 		d.append((l[0], l[1], l[2]))
# 	return d

if __name__=='__main__':
	nlp = English()
	s = fetch_content_from_doc('april_output')
	doc = nlp(s)

	sents = filter_sentences(doc, 'May')
	for s in sents:
		print(s)