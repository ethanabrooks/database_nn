import xml.sax
import mwparserfromhell

# Takes in a file where each line is in the form: entity, answer
# returns a dict where the key is the entity, and the value is the answer
# Assumes no extra space included in the file
# IGNORES ALL CASES
def read_entity_answer_pair(filename):
	with open(filename) as f:
		lines = f.read().splitlines()
	d = {}
	for line in lines:
		l = line.split(',')
		d[l[0].lower()] = l[1].lower()
	return d


# Class that extends ContentHandler
# matched_output is a list of tuples (entity name, answer, wiki text)
# IGNORES ALL CASES
class WikiContentHandler(xml.sax.ContentHandler):
	def __init__(self, entity_pairs):
		xml.sax.ContentHandler.__init__(self)
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

		# List of outputs
		self.matched_output = []

	# Removes wikipedia markup using external library
	@staticmethod
	def clean_markup(content):
		clean_code = mwparserfromhell.parse(content).strip_code()
		return clean_code

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
				self.matched_output.append(
					(self.current_entity, 
					self.entity_pairs[self.current_entity], 
					self.clean_markup(self.current_entity_text))
				)
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
			if title in self.entity_pairs:
				self.entity_flag = True
				self.current_entity = title
		elif self.text_flag:
			if self.entity_flag:
				self.current_entity_text += content


if __name__=='__main__':
	d = read_entity_answer_pair('entity_answer_pair.txt')

	source = open('simplewiki-20160203-pages-meta-current.xml')
	handler = WikiContentHandler(d)
	xml.sax.parse(source, handler)

	l = handler.matched_output

	output_filename = 'output'
	output = open(output_filename, 'w')

	for x in l:
		output.write(x[0] + ',' + x[1] + ',' + x[2] + '\n')

	source.close()