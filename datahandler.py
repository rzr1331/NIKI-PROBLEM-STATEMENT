# Data Handler and Text Cleaner

import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

class DataHandler:
  
    def __init__(self, questionDataSet) :
        self.questionDataSet = questionDataSet
        
    def cleanStemmer(self) :
        ps = PorterStemmer()
        corpus = []
        for i in range(0,len(self.questionDataSet.index)):
            question = re.sub('[^a-zA-Z]', ' ', self.questionDataSet[i])
            question = question.lower()
            question = question.split()  
            question = [ps.stem(word) for word in question]
            question = ' '.join(question)
            corpus.append(question)
        return corpus
    
    def cleanLemmatizer(self) :
        lm = WordNetLemmatizer()
        corpus = []
        for i in range(0,len(self.questionDataSet.index)):
            question = re.sub('[^a-zA-Z]', ' ', self.questionDataSet[i])
            question = question.lower()
            question = question.split()  
            question = [lm.lemmatize(word) for word in question]
            question = ' '.join(question)
            corpus.append(question)
        return corpus
    
        
