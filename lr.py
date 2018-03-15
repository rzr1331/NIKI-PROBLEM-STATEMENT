# Logistic Regression

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

class LR :
    def __init__(self, X_train, y_train) :
        # Fitting Logistic Regression to the Training set
        self.classifier = LogisticRegression(random_state = 0)
        self.classifier.fit(X_train,y_train)

    def predictor(self, X_test) :
        # Predicting the Test set results
        y_pred = self.classifier.predict(X_test)
        return y_pred

    def accuracy(self, y_test, y_pred) :
        # Making the Confusion Matrix & calculating Accuracy
        cm = confusion_matrix(y_test, y_pred)
        accuracy = np.trace(cm)/np.sum(cm)
        return accuracy*100
   