import nltk								# Import nltk. See www.nltk.org/install.html
from nltk.corpus import wordnet as wn   # Import WordNet 

def main():								# Define main function
	word1 = input('Word1:')				# Get input from terminal
	word2 = input('Word2:')
	print(cal_similarity(word1, word2))	# Call 'noun_similarity' function and print result

def cal_similarity(Word1, Word2):		# Define 'noun_similarity', 'Word1' and 'Word2' are the arguments
	word1 = wn.synsets(Word1)			# Get all of the word synsets, by using wn.synsets
	word2 = wn.synsets(Word2)			# and get wordnet synsets List

	highest_score = 0					# Initialize highest_score
	highest_pair = []					# Initialize highest_pair list

	for i in word1:
		for j in word2:
			score = i.path_similarity(j)	# Calculate the similarity between two synsets
			if score == None:				# If similarity is None, skip to next round
				continue
			elif float(score) > highest_score:
				highest_score = score
				highest_pair = i, j

	print('Word1:{0} total {1} synset(s)'.format(Word1, len(word1)))
	print('Word2:{0} total {1} synset(s)'.format(Word2, len(word2)))

	return highest_pair, highest_score

if __name__ == '__main__':
	main()