
from gensim.models import KeyedVectors
pathToBinVectors = 'C:\\Users\DE104752\\Documents\\word2vec_pretrained\\google.bin'

print("Loading the data file... Please wait...")
MODEL = KeyedVectors.load_word2vec_format(pathToBinVectors, binary=True)
print("Successfully loaded file!")

# How to call one word vector?
# MODEL['resume'] -> This will return NumPy vector of the word "resume".

import numpy as np
import math
from nltk.corpus import stopwords


class PhraseVector:
	def __init__(self, phrase):
		self.vector = self.PhraseToVec(phrase)
	# Calculates similarity between two sentences (= two  sets of vectors) based on the averages of the sets.
	#"ignore"  = "The vectors within the set that need to be ignored. If this is an empty list, nothing is ignored. 
	# returns the condensed single vector that has the same dimensionality as the other vectors within the vecotSet
	def ConvertVectorSetToVecAverageBased(self, vectorSet, ignore = []):
		if len(ignore) == 0: 
			return np.mean(vectorSet, axis = 0)
		else: 
			return np.dot(np.transpose(vectorSet),ignore)/sum(ignore)

	def PhraseToVec(self, phrase):
		cachedStopWords = stopwords.words("english")
		phrase = phrase.lower()
		wordsInPhrase = [word for word in phrase.split() if word not in cachedStopWords]
		vectorSet = []
		for aWord in wordsInPhrase:
			try:
				wordVector=MODEL[aWord]
				vectorSet.append(wordVector)
			except:
				pass
		return self.ConvertVectorSetToVecAverageBased(vectorSet)

	def CosineSimilarity(self, otherPhraseVec):
		cosine_similarity = np.dot(self.vector, otherPhraseVec) / (np.linalg.norm(self.vector) * np.linalg.norm(otherPhraseVec))
		try:
			if math.isnan(cosine_similarity):
				cosine_similarity=0
		except:
			cosine_similarity=0		
		return cosine_similarity


if __name__ == "__main__":
	print(""
       ######## WELCOME TO THE PHRASE SIMILARITY CALCULATOR ############
       "")
	while True:
		userInput1 = input("Type the phrase1: \n")
		userInput2 = input("Type the phrase2: \n")

		phraseVector1 = PhraseVector(userInput1)
		phraseVector2 = PhraseVector(userInput2)
		similarityScore  = phraseVector1.CosineSimilarity(phraseVector2.vector)

		print("Similarity Score: {} ".format(similarityScore))