import os
import io

directory = '../Attentive_Reader/deepmind-qa/cnn/questions/'

if __name__ == '__main__':
	all_data_sets = []
	# all_data_sets.append(directory + 'training/')
	# all_data_sets.append(directory + 'test/')
	all_data_sets.append(directory + 'validation/')
	total_ents = 0
	total_docs = 0
	for data_set in all_data_sets:
		for i in os.listdir(data_set):
			f = open(data_set + i, 'r')
			lines = f.read().splitlines()
			total_ents += len(lines[8:])
			total_docs += 1
			f.close()

	print('total ents: ' + str(total_ents))
	print('total docs: ' + str(total_docs))


# total ents: 428041
# total docs: 44072