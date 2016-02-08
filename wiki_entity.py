from spacy.en import English

def get_ents_from_string(sentence):
	tokens = nlp(sentence)
	return tokens.ents

if __name__=='__main__':
	nlp = English(parser=False,tagger=False)
	sentence = 'Tom went to give a banana to the girl Suzy.'
	ents = get_ents_from_string(sentence)
	for ent in ents:
		print((ent, ent.label_))