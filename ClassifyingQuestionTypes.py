# Natural Language Processing

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
from sklearn.preprocessing import LabelEncoder
dataset = pd.read_csv('LabelledData.txt', delimiter = ',,,', quoting = 3, header=None)
y = dataset.iloc[:,1].str.strip()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Cleaning the texts
import re
import nltk
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1483):
    question = re.sub('[^a-zA-Z]', ' ', dataset[0][i])
    question = question.lower()
    question = question.split()
    ps = PorterStemmer()
    question = [ps.stem(word) for word in question]
    question = ' '.join(question)
    corpus.append(question)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix & calculating Accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = np.trace(cm)/np.sum(cm)
print("Accuracy : ", accuracy*100)

##############################################################
# For Testing New Sentences :
##############################################################
check_question = "What actor starred in 1980 's Blue Lagoon , 1982 's The Pirate Movie and 1983 's A Night in Heaven ?"
check_question = re.sub('[^a-zA-Z]', ' ', check_question)
check_question = check_question.lower()
check_question = check_question.split()
ps = PorterStemmer()
check_question = [ps.stem(word) for word in check_question]
check_question = ' '.join(check_question)
local_corpus = [check_question]

Check_X = cv.transform(local_corpus)
Check_X = Check_X.toarray()

Check_y = classifier.predict(Check_X)
print("Check Question Type : ",labelencoder_y.inverse_transform(Check_y))