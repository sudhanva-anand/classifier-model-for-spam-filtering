# Import the required libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import string
import matplotlib.pyplot as plt
# Load the dataset

data = pd.read_csv("spam.tsv",sep='\t',names=['Class','Message'])
data.head(2) # View the first 8 records of our dataset
# Summary of the dataset
data.info()
# create a column to keep the count of the characters present in each record
data['Length'] = data['Message'].apply(len)
# view the dataset with the column 'Length' which contains the number of characters present in each mail
data.head(5)
# statistical info of the data
data.describe()
# Let's see the count of each class
data['Class'].value_counts()
# Lets assign ham as 1
data.loc[data['Class']=="ham","Class"] = 1
# Lets assign spam as 0
data.loc[data['Class']=="spam","Class"] = 0
data.head(8)
# Why is it important to remove punctuation?

"This message is spam" == "This message is spam."
# get the default list of punctuations in Python
import string

string.punctuation
# Creating a function to remove the punctuation

def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation]) 
    return text
s = "data// science!!"
remove_punct(s)
text = []
for i in data['Message']:
    t = remove_punct(i)
    text.append(t)

    data['Text_clean'] = text
data
# creating new column 'text_clean' to hold the cleaned text

data['text_clean'] = data['Message'].apply(lambda x: remove_punct(x)) #the lambda keyword is used to create anonymous functions
                                                                       
# view the dataset
data.head()
# Splitting x and y

X = data['text_clean'].values # convert df as array
y = data['Class'].values

X
# Datatype for y is object. lets convert it into int
y = y.astype('int')
y
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=10)
X_train.shape
X_test.shape
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer



# Initialize the object for countvectorizer 
CV = CountVectorizer(stop_words="english")  
# Apply countvectorizer functionality on the training data to convert 
# the categorical data into vectors
X_train_CV = CV.fit_transform(X_train)

#to ignore warnings
import warnings
warnings.filterwarnings('ignore')

CV.get_feature_names()
# Initialising the model
NB = MultinomialNB()
# feed data to the model
#NB.fit(xSet_train_CV,ySet_train)
NB.fit(X_train_CV,y_train)
# Let's apply CV on our test data. 
X_test_CV = CV.transform(X_test) #transform() is used to avoid the data leakage
# prediction for xSet_test_CV

y_predict = NB.predict(X_test_CV)
y_predict
# classification report

print(classification_report(y_test,y_predict))
## confusion matrix
pd.crosstab(y_test,y_predict)
#Initialising a model
bnb = BernoulliNB()

## fitting the model
bnb.fit(X_train_CV,y_train)

## getting the prediction
y_hat1=bnb.predict(X_test_CV) 

## confusion matrix
pd.crosstab(y_test,y_hat1)
# Splitting x and y

X = data['text_clean'].values
y = data['Class'].values
y
# Datatype for y is object. lets convert it into int
y = y.astype('int')
y
# Split the data into training and testing
# convert the training data - fit_transform()
# convert the testing data - transform()
## text preprocessing and feature vectorizer
# To extract features from a document of words, we import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


tf=TfidfVectorizer() ## object creation

#X=tf.fit_transform(X) ## fitting and transforming the data into vectors
## Creating training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=6)
## print feature names selected by TDIDF from the raw documents
#tf.get_feature_names()
## number of features created
#len(tf.get_feature_names())
X_train_cv = tf.fit_transform(X_train)
X_test_cv = tf.transform(X_test)
# Initialising the model
nb = MultinomialNB()
nb.fit(X_train_cv,y_train)  
    y_hat = nb.predict(X_test_cv)
# classification report

print(classification_report(y_test,y_hat))
pd.crosstab(y_test,y_hat)
## model object creation
nb=BernoulliNB()

## fitting the model
nb.fit(X_train_cv,y_train)

## getting the prediction
y_hat=nb.predict(X_test_cv)
y_hat
## Evaluating the model
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_hat))
## confusion matrix
pd.crosstab(y_test,y_hat)