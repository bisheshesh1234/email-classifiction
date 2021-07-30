# email-classifiction
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayse import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

##step1: load dataset
dataframe = pd.read_csv("spam.svm")
print(dataframe.head())
##step2: Split into Training and Test Data
x=dataframe["EmailText"];
y=dataframe["Label"];
x_train, y_train = x[0:4457],y[0:4457]
x_test, y_test = x[4457:], y[4457:]

##step3: Extract Feature
cv=CountVectorizer()
features = cv.fit_transform(x_train)
##step4: Build Model
turned_parameters = {'kernel':['linear','rbf'],'gamma':[1e-3,1e-4],'C':[1,10,100,1000]}
model = GridSearch(svm.SVC(),tuned_parameters)
model.fit(features,y_train)
print(model.best_params)
##step5: Test Accuracy
features_test = cv.transform(x_test)
print("Accuracy of the model is:",model.score(features_test,y_test)
