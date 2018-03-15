# Data Handler and Text Cleaner

import re
from nltk.stem.porter import PorterStemmer

class DataHandler:
  
    def __init__(self, questionDataSet) :
        self.questionDataSet = questionDataSet
        
    def cleanStemmer(self) :
        corpus = []
        for i in range(0,len(self.questionDataSet.index)):
            question = re.sub('[^a-zA-Z]', ' ', self.questionDataSet[i])
            question = question.lower()
            question = question.split()  
            ps = PorterStemmer()
            question = [ps.stem(word) for word in question]
            question = ' '.join(question)
            corpus.append(question)
        return corpus
    
        
