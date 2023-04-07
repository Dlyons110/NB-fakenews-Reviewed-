
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 11:35:19 2023
@author: Dennis
"""

# Import the relevant packages
import pandas as pd
from pandas.api.types import CategoricalDtype
from IPython.display import display, Markdown
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


# id: unique id for a news article
# title: the title of a news article
# author: author of the news article
# text: the text of the article; could be incomplete
# label: a label that marks the article as potentially unreliable
# 0: reliable
# 1: unreliable



# Import and print the dataset - Loading the dataset might take some time because of the internet connection
rawDF = pd_read_csv ('https://raw.githubusercontent.com/HAN-M3DM-Data-Mining/assignments/master/datasets/NB-fakenews.csv')
rawDF
       

# print the column names
print(rawDf.columns)
# Drop unnecessary columns
cleanDF = rawDF.drop (['id', 'title','author'], axis = 1)

# define a dictionary to map label values to categories
label_map = {0: 'reliable', 1: 'unreliable'}

# create a new DataFrame with updated 'categorie' column
cleanDF2 = cleanDF.assign(category=cleanDF[author].map(label_map))

# print the updated DataFrame
print(cleanDF2)



# Categorise the data as "reliable" or "unreliable" and print this outcome
catType = CategoricalDtype(categories=["reliable", "unreliable"], ordered=False)

cleanDF2.category = cleanDF2.category.astype(catType)
cleanDF2.category


# Print the distribution of the messages using count
cleanDF2.category.value_counts()
# Print the same distribution but using relative numbers
cleanDF2.category.value_counts(normalize=True)



# Generate a word cloud image
## Note to teacher: used str to convert text
reliableText = ' '.join([str(Text) for Text in cleanDF2[cleanDF2['category']=='reliable']['text'].astype(str)])
unreliableText = ' '.join([str(Text) for Text in cleanDF2[cleanDF2['category']=='unreliable']['text'].astype(str)])

colorListReliable=['#e9f6fb','#92d2ed','#2195c5']
colorListUnreliable=['#f9ebeb','#d57676','#b03636']
colormapReliable=colors.ListedColormap(colorListReliable)
colormapUnreliable=colors.ListedColormap(colorListUnreliable)
wordcloudReliable = WordCloud(background_color='white', colormap=colormapReliable).generate(reliableText)
wordcloudUnreliable = WordCloud(background_color='white', colormap=colormapUnreliable).generate(unreliableText)

# Display the generated image (run all the lines below at once):
fig, (wc1, wc2) = plt.subplots(1, 2)
fig.suptitle('Wordclouds for reliable and unreliable')
wc1.imshow(wordcloudReliable)
wc2.imshow(wordcloudUnreliable)
plt.show()




### --------- Training Model

# Remove the rows that contain NaN values 
## Note to teacher: dropped the nan values
rawDF.dropna(subset=[text], inplace=True)

# Convert text data into a numerical vector 
vectorizer = TfidfVectorizer(max_features=1000)
vectors = vectorizer.fit_transform(rawDF.text)
wordsDF = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names_out())
wordsDF.head()

# Split the dataset in four sets 
xTrain, xTest, yTrain, yTest = train_test_split(wordsDF, rawDF.label)


# use algorithm to train a model on a dataset to predict new data
bayes = MultinomialNB()
bayes.fit(xTrain, yTrain)

# use trained model (bayes) to predict the labels of the test dataset and compare the predicted outcome
yPred = bayes.predict(xTest)
yTrue = yTest

# calculate the accuracy of a classification model, print the score, generate a confusion matrix based on the prediction and display the matrix
accuracyScore = accuracy_score(yTrue, yPred)
print_(f'Accuracy: {accuracyScore}')
matrix = confusion_matrix(yTrue, yPred)
labelNames = pd.Series(['reliable', 'unreliable'])
pd.DataFrame(matrix, columns='Predicted ' + labelNames, index='Is ' + labelNames)

#Problems with Sabotaged Version
Authors name is incorrect 
Reviewer Code line was not included 
Line 38 RawDF has no linkage within the code 
Line 62 and 61 are just duplicates of code but 62 having no input on the code
Unnesecary need of line space between line 84 and 89
Unnesecary need of line space between line 63 and 66