import io

NUM_FILES = 50
DEFAULT_PATH = "jeopardy_question/question_answer_pairs_"

if __name__ == '__main__':
	questions = open('question_answer_pairs', 'r')

	files = []
	for i in range(NUM_FILES):
		files.append(open(DEFAULT_PATH + str(i), 'w+'))

	lines = questions.read().splitlines()
	i = 0
	for line in lines:
		files[i].write(line + '\n')
		i = (i + 1) % 50