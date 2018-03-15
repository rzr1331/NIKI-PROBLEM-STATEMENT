# Natural Language Processing

# Importing the libraries
import numpy as np
import pandas as pd
import re
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from datahandler import DataHandler
from lr import LR
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Importing the dataset
from sklearn.preprocessing import LabelEncoder
dataset = pd.read_csv('LabelledData.txt', delimiter = ',,,', quoting = 3, header=None, engine='python')
y = dataset.iloc[:,1].str.strip()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Cleaning the texts
corpus = []
cleaner = DataHandler(dataset.iloc[:,0])
#print(cleaner.__dict__)
corpus = cleaner.cleanStemmer()
    
# Creating the Bag of Words model
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

##############################################################
# Training the model
print("Training the model with train:test :: 80:20")

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Logistic Regression to the Training set
model = LR(X_train,y_train)
y_pred = model.predictor(X_test)

# Calculating Accuracy
accuracy = model.accuracy(y_test, y_pred)
print("Model Accuracy : ", accuracy)

# Retraining the model with complete dataset
model = LR(X,y)
print("Model Retrained With the Complete DataSet")

##############################################################
# For Testing New Sentences :
##############################################################

while True:
    test_question = input("Enter A Question to find the category : (Leave Blank to exit)")
    if test_question == "":
        break
    else :
        question = pd.Series([test_question])
        cleaner2 = DataHandler(question)
        Corpus_question = cleaner2.cleanStemmer()
        X_question = cv.transform(Corpus_question).toarray()
        y_question = model.predictor(X_question)
        print("Check Question Type : ",labelencoder_y.inverse_transform(y_question))
