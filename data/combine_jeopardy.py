import os
import io

directory = 'jeopardy_results/'

if __name__ == '__main__':
	output = open('nn_output_combined', 'w+')
	for i in os.listdir(directory):
		f = open(directory + i, 'r')
		lines = f.read().splitlines()
		for line in lines:
			output.write(line + '\n')


