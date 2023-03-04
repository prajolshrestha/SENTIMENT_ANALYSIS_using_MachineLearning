"""
SENTIMENT ANALYSIS USING NATURAL LANGUAGE PROCESSING WITH DIFFERENT ML MODELS
A: NAIVE BAYES
B: SVM
C: Kernel SVM
D: Decision Tree
E: Random Forest
F: Logistic Regression
G: KNN
"""

#a)import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#b)import dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting= 3)  #qouting to remove ""
#print(dataset)



#c)cleaning texts
import re   #Regular expression operations
import nltk #Natural Language Toolkit
nltk.download('stopwords')

# a corpus or text corpus is a language resource consisting of a large and structured set of texts. 
# In corpus linguistics, they are used to do statistical analysis and hypothesis testing, checking 
# occurrences or validating linguistic rules within a specific language territory
from nltk.corpus import stopwords 


#An algorithm for suffix stripping
#Interfaces used to remove morphological affixes from words, leaving only the word stem. 
# Stemming algorithms aim to remove those affixes required for eg. grammatical role, tense, 
# derivational morphology leaving only the stem of the word.
from nltk.stem.porter import PorterStemmer



#lets create empty storage to add text and make a corpus
corpus = []

#lets do cleaning for each reviews
for i in range(0,1000):  #we have 1000 reviews in our dataset
    
    #now lets clean all str other than letters
    review = re.sub('[^a-zA-Z]',' ', dataset.iloc[i,0]) #replace letters other than a-z/A-Z by a space
    review = review.lower()  #change all letters to lower case
    review = review.split()  #Split a string into a list where each word is a list item
    #till now we have a list of words in "review" variable
    
    #now lets apply stemming algorithm
    ps = PorterStemmer() #create object of stem algo
    all_stopwords = stopwords.words('english') #list of all stopwords of english language
    all_stopwords.remove('not')    #type = list
    #a = set(all_stopwords)    #type= set
    review = [ps.stem(word) for word in review if not word in all_stopwords] #type=list
    #join elements of list using space
    review = ' '.join(review)  #type=str 

    corpus.append(review) #appends each (1000) review in 'corpus' variable after seperating



#c)lets create bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray() #Convert a collection of text documents to a matrix of token counts.
y = dataset.iloc[:,-1].values
#print(len(x[0]))
#print(x)
#print(y)


#d)split data for train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
#print(x_train)


###########################################e)training and predict#####################################
# logistic regression
def logistic_regression():
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    return y_pred

def knn():
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5,p=2, metric='minkowski')
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    return y_pred

#a. Naive Bayes
def naive_bayes():
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    return y_pred

def svm():
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    return y_pred
    
def kernel_svm():
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    return y_pred

#b. Decision Tree
def decision_tree():
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    return y_pred

#c. Random Forest
def random_forest():
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    return y_pred

y_pred = [logistic_regression(),knn(),naive_bayes(),svm(),kernel_svm(),decision_tree(),random_forest()]

#############################################f)Analyzing results#############################################
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, accuracy_score, f1_score,precision_score,recall_score
con_matrix = []
accuracy_score_values =[]
f1_score_values=[]
precision_score_values = []
recall_score_values = []
for i in range(0,len(y_pred)):
    cm = confusion_matrix(y_test,y_pred[i])
    # cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classifier.classes_)
    # cm_disp.plot()
    # plt.show()
    
    con_matrix.append(cm)
    accuracy_score_values.append((accuracy_score(y_test,y_pred[i])))
    f1_score_values.append(f1_score(y_test,y_pred[i]))
    precision_score_values.append(precision_score(y_test,y_pred[i]))
    recall_score_values.append(recall_score(y_test,y_pred[i]))
# print(con_matrix)
# print(accuracy_score_values)
# print(f1_score_values)

#make a dataframe
classifier_name = ['logistic_regression','knn','naive_bayes','svm','kernel_svm','decision_tree','random_forest']
df = pd.DataFrame([classifier_name,con_matrix,accuracy_score_values,precision_score_values,recall_score_values,f1_score_values]).transpose() 
df.columns = ['classifier name','confusion matrix','accuracy score','precision score','recall score', 'f1 score']
print(df)



########################################g) new prediction #########################################################
while(True):
    new_review = input('Enter your review:')

    #cleaning
    new_review = re.sub('[^a-zA-Z]',' ',new_review) #not all letters a-z and A-Z are replaced by space
    new_review = new_review.lower()
    new_review = new_review.split()
        
    #stemming
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]

    #bag of words
    x_test = cv.transform(new_corpus).toarray()

    #predict using ML model
    new_y_pred = svm()

    #display results
    if new_y_pred == 1:
        print('Thankyou for your positive response!')

    else:
        print('We are sorry for your inconvinance!')

    print('To end review click control + c')    

