# SENTIMENTANALYSIS

#Sentiment Analysis using Natural Language Processing with Different ML Models
This is a project to perform sentiment analysis on a dataset of restaurant reviews using natural language processing with different machine learning models. 
This project uses seven different models:

Logistic Regression
KNN
Naive Bayes
SVM
Kernel SVM
Decision Tree
Random Forest

#Libraries Used
This project uses the following libraries:

pandas
matplotlib
numpy
re
nltk
sklearn

#Dataset
The dataset used in this project is the Restaurant_Reviews.tsv file. It contains 1000 restaurant reviews, with each review being labeled as either positive or negative.

#Cleaning Texts
The texts in the dataset are cleaned using regular expression operations and the Natural Language Toolkit (nltk). All non-letter characters are removed and all letters are converted to lowercase. Stopwords are also removed from the reviews. The resulting cleaned texts are then used to create a bag of words model.

#Bag of Words Model
The bag of words model is created using the CountVectorizer class from the sklearn.feature_extraction.text library. The resulting model has 1500 features. The reviews are then split into training and testing sets.

#Machine Learning Models
The seven machine learning models are trained and tested using the training and testing sets. The accuracy, f1-score, precision, and recall are calculated for each model.

#Results
The following are the accuracy, f1-score, precision, and recall for each of the seven machine learning models:

Model	Accuracy	F1-Score	Precision	Recall
Logistic Regression	73.50%	0.72	0.76	0.69
KNN	63.00%	0.61	0.68	0.56
Naive Bayes	71.50%	0.70	0.73	0.67
SVM	74.50%	0.73	0.78	0.69
Kernel SVM	76.50%	0.75	0.81	0.70
Decision Tree	66.50%	0.65	0.69	0.62
Random Forest	73.50%	0.72	0.76	0.69

From the results, it can be seen that the **Kernel SVM model** has the highest accuracy, f1-score, precision, and recall.

#Prediction
It uses a machine learning (ML) model to predict whether a user's response to a prompt is positive or negative. Here's what it does:

new_y_pred = svm() calls a function called svm() that presumably uses a Support Vector Machine (SVM) algorithm to make the prediction. The function likely takes some input data and returns a prediction (1 for positive, 0 for negative). ## use any ML Model to predict

if new_y_pred == 1: checks if the prediction is positive. If it is, the script prints 'Thankyou for your positive response!'.
else: executes if the prediction is negative. It prints 'We are sorry for your inconvinance!'.

Finally, the script prints 'To end review click control + c'. This is a message to the user indicating how to stop the script from running (by pressing Control + C on their keyboard).

Overall, this script appears to be designed to automatically classify user responses and provide appropriate feedback, presumably to assist in some sort of customer service or feedback collection process.

#Credits
This motion detector app was built by **PRAJOL SHRESTHA** as a personal project. If you have any feedback or suggestions, feel free to create a pull request or contact me via email.

#Licence
This motion detector app is licensed under the MIT License. You are free to use, modify, and distribute this application as long as you give credit to the original author.
