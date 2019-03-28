#--------------------------------------------------Libraries Required----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
#---------------------------------------------------Importing dataset---------------------------------------
dataset= pd.read_csv("train_data.csv")
dataset1 = pd.read_csv("test_data.csv")
df = dataset.append(dataset1,ignore_index=True)
df = df[['tweet','label']]


#-------------------------------------------------Cleaning the tweets------------------------------------
Clean_tweets = []
for i in range(0, 9873):
    Tweet = df['tweet'][i]
#Since "$&@*#" denote most profane and vulgar terms So, keeping those words replacing with "Vulgar"  
    Tweet = Tweet.replace("$&@*#","Vulgar")
#Removing all the '@tagged','urls', punctuations, special characters etc which do not add up to value in analysis
    Tweet =re.sub('(@[\w]+)|(\S+\.\S+\s?)|(http\S+)|[^a-zA-Z#]',' ',Tweet)
#Converting all the text to lower case    
    Tweet = Tweet.lower()
#Splitting whole text to list of words
    Tweet = Tweet.split()
#Removing all the words of length below 3 like 'us' etc which donot seem like adding value
    Tweet = [word for word in Tweet if len(word)>=2 ]
#Stemming or removing all the suffix to keep only root words 
    ps = PorterStemmer()
    Tweet = [ps.stem(word) for word in Tweet if not word in set(stopwords.words('english'))]
#Removing repeated words and keeping unique set of words in the text    
    Tweet = ' '.join(sorted(set(Tweet),key= Tweet.index))
    Clean_tweets.append(Tweet)

df['Cleaned_tweets'] = Clean_tweets

#Removing the most frequent words that does not value for analysis
Overall_freq = pd.Series(''.join(df['Cleaned_tweets']).split()).value_counts()[0:4]
Overall_freq

Overall_freq = list(Overall_freq.index)
df['Cleaned_tweets'] = df['Cleaned_tweets'].apply(lambda x: " ".join(x for x in x.split() if x not in Overall_freq))

Overall_freq1 = pd.Series(''.join(df['Cleaned_tweets']).split()).value_counts()[3:8]
Overall_freq1
Overall_freq1 = list(Overall_freq1.index)
df['Cleaned_tweets'] = df['Cleaned_tweets'].apply(lambda x: " ".join(x for x in x.split() if x not in Overall_freq1))

#Removing least frequent words that does not give value
Least_freq = pd.Series(''.join(df['Cleaned_tweets']).split()).value_counts()[-20:] 
Least_freq
Least_freq = list(Least_freq.index)
df['Cleaned_tweets'] = df['Cleaned_tweets'].apply(lambda x: ' '.join([x for x in x.split() if x not in Least_freq]))


#-------------------------------------------------Visualizing the Words-----------------------------------
from wordcloud import WordCloud

#All the words
Complete_words = ' '.join([word for word in df['Cleaned_tweets']])
wordcloud = WordCloud(width=700, height=600, random_state=2, max_font_size=100).generate(Complete_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#Positive words cloud
Positive_words = ' '.join([word for word in df['Cleaned_tweets'][df['label']==0]])
wordcloud = WordCloud(width=700, height=600, random_state=2, max_font_size=100).generate(Positive_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#Negative Words cloud
Negative_words = ' '.join([word for word in df['Cleaned_tweets'][df['label']==1]])
wordcloud = WordCloud(width=700, height=600, random_state=2, max_font_size=100).generate(Negative_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#Function for collecting Hashtags
def collect_hashtags(x):
    Hashtags = []
    for i in x:
       HT= re.findall(r"#(\w+)",i)
       Hashtags.append(HT)
    return Hashtags;

Positive_hashtags = collect_hashtags(df['Cleaned_tweets'][df['label']==0])
Negative_hashtags = collect_hashtags(df['Cleaned_tweets'][df['label']==1])

#WordCloud of positive hashtags to visualize the most frequent words 
Positive_hashtags = ' '.join(map(str,[Positive_hashtags]))
wordcloud = WordCloud(width=700, height=600, random_state=2, max_font_size=100).generate(Positive_hashtags)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#WordCloud of Negative hashtags to Visualize the most frequent words
Negative_hashtags = ' '.join(map(str,[Negative_hashtags]))
wordcloud = WordCloud(width=700, height=600, random_state=2, max_font_size=100).generate(Negative_hashtags)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#Unnesting the lists
P_hashtags = sum(Positive_hashtags,[])
N_hashtags = sum(Negative_hashtags,[])
#Impact of positive hashtags
P = nltk.FreqDist(P_hashtags)
d = pd.DataFrame({'Hashtags': list(P.keys()),
                  'count': list(P.values())})
d = d.nlargest(columns='count',n=20)
plt.figure(figsize = (16,5))
ax = sns.barplot(data = d,x = 'Hashtags',y = 'count')
ax.set(ylabel = "Count")
plt.show()

#Impact of negative hashtags
N = nltk.FreqDist(N_hashtags)
d = pd.DataFrame({'Hashtags': list(N.keys()),
                  'count': list(N.values())})
d = d.nlargest(columns='count',n=20)
plt.figure(figsize = (16,5))
ax = sns.barplot(data = d,x = 'Hashtags',y = 'count')
ax.set(ylabel = "Count")
plt.show()

# Since we got to know that it is good to include hashtag words lets remove '#' and keep words
df['Cleaned_tweets'] = df['Cleaned_tweets'].apply(lambda x: re.sub('[#]','',x))

#Knowing the sentimnent of each text in the data including this feature would boost the model prediction
df['Sentiment'] = df['Cleaned_tweets'].apply(lambda x: TextBlob(x).sentiment[0])

#Average Word Length
length = []
for i in range(0,9873):
    wo = df['Cleaned_tweets'][i].split()
    wo = sum(len(word) for word in wo)/len(wo)
    length.append(wo);

df['Avg_word_Length'] = length
#Now removing the previous tweet column from the data
df.drop(columns = 'tweet',inplace=True)

#Features extraction from cleaned_tweets for building models
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn_pandas import DataFrameMapper


#---------------------Bag of words features with added Sentiment and Average word length features----------- 
cv = CountVectorizer(max_features = 10000)
mapper = DataFrameMapper([
     ('Cleaned_tweets', cv),
     ('Sentiment', None),
     ('Avg_word_Length', None),
 ])
BOW_train = mapper.fit_transform(df)

#Term Frequency-Inverse Document Frequency (Tf-Idf) features with added Sentiment and Avg word length features
Tfidf = TfidfVectorizer(max_features=10000)
mapper = DataFrameMapper([
     ('Cleaned_tweets', Tfidf),
     ('Sentiment', None),
     ('Avg_word_Length', None),
 ])
Tfidf_train = mapper.fit_transform(df)

#-------------------------------------------Model Building---------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
#---------------------------------------------------------Logistic Regression-------------------------------------------
from sklearn.linear_model import LogisticRegression
#----------------------------------------------Bag of Words model-----------------------------------------
Train_BOW = BOW_train[:7920,:]
Test_BOW = BOW_train[7920:,:]
Y = dataset.iloc[:,1].values

Xtrain_BOW,Xtest_BOW,Ytrain_BOW,Ytest_BOW = train_test_split(Train_BOW, Y, test_size =0.25, random_state=0) 

Lregressor_BOW = LogisticRegression()
Lregressor_BOW.fit(Xtrain_BOW,Ytrain_BOW)

y_pred_BOW_LR = Lregressor_BOW.predict(Xtest_BOW)

from sklearn.metrics import confusion_matrix
cm_BOW_LR = confusion_matrix(Ytest_BOW, y_pred_BOW_LR)

f1_score(Ytest_BOW,y_pred_BOW_LR)

#Predictions for Test set 
Test_pred_LR = Lregressor_BOW.predict(Test_BOW)
dataset1['label'] = Test_pred_LR
LR_submission = dataset1[['id','label']]
LR_submission.to_csv("LR_submission_0.75.csv") 

#---------------------------------------------Tfidf Model---------------------------------------------------
Train_Tfidf = Tfidf_train[:7920,:]
Test_Tfidf = Tfidf_train[7920:,:] 

Xtrain_Tfidf,Xtest_Tfidf,Ytrain_Tfidf,Ytest_Tfidf = train_test_split(Train_Tfidf, Y, test_size =0.25, random_state=0) 

Lregressor_Tfidf = LogisticRegression()
Lregressor_Tfidf.fit(Xtrain_Tfidf,Ytrain_Tfidf)
y_pred_Tfidf_LR = Lregressor_Tfidf.predict(Xtest_Tfidf)

#Confusion matrix error checking
cm_Tfidf_LR = confusion_matrix(Ytest_Tfidf, y_pred_Tfidf_LR)
#F-1 Score validation
f1_score(Ytest_Tfidf,y_pred_Tfidf_LR)

#Predictions for Test set 
Test_pred_Tfidf = Lregressor_Tfidf.predict(Test_Tfidf)
dataset1['label'] = Test_pred_Tfidf
Tfidf_submission = dataset1[['id','label']]
Tfidf_submission.to_csv("Tfidf_submission_0.69.csv") 
#---------------------------------------------------Random Forest Classification------------------------------
from sklearn.ensemble import RandomForestClassifier
#---------------------------------------Bag of words Model-------------------------------------------------
classifier_RFR_BOW = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier_RFR_BOW.fit(Xtrain_BOW, Ytrain_BOW)
y_pred_RFR_BOW = classifier_RFR_BOW.predict(Xtest_BOW)

#Confusion matrix error checking
cm_BOW_RFR = confusion_matrix(Ytest_BOW, y_pred_RFR_BOW)
#F-1 Score validation
f1_score(Ytest_BOW,y_pred_RFR_BOW)

#Predictions for Test set
Test_pred_RFR_BOW = classifier_RFR_BOW.predict(Test_BOW)
dataset1['label'] = Test_pred_RFR_BOW
RFR_submission_BOW = dataset1[['id','label']]
RFR_submission_BOW.to_csv("RFR_submission_BOW_0.73.csv") 

#----------------------------------------Tfidf model-----------------------------------------------
classifier_RFR_Tfidf = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
classifier_RFR_Tfidf.fit(Xtrain_Tfidf, Ytrain_Tfidf)
y_pred_RFR_Tfidf = classifier_RFR_Tfidf.predict(Xtest_Tfidf)

#Confusion matrix error checking
cm_Tfidf_RFR = confusion_matrix(Ytest_Tfidf, y_pred_RFR_Tfidf)
#F-1 Score validation
f1_score(Ytest_Tfidf,y_pred_RFR_Tfidf)

#Predictions for Test set
Test_pred_RFR_Tfidf = classifier_RFR_Tfidf.predict(Test_Tfidf)
dataset1['label'] = Test_pred_RFR_Tfidf
RFR_submission_Tfidf = dataset1[['id','label']]
RFR_submission_Tfidf.to_csv("RFR_submission_Tfidf_0.73.csv") 

#----------------------------------------------------SUpport Vector Machine--------------------------------------------
#----------------------------------------Bag of Words----------------------------------------------------
from sklearn.svm import SVC
classifier_BOW_SVM = SVC(kernel = 'linear', random_state = 0)
classifier_BOW_SVM.fit(Xtrain_BOW, Ytrain_BOW)
y_pred_SVM_BOW = classifier_BOW_SVM.predict(Xtest_BOW)

#Confusion Matrix for error checking
cm_SVM_BOW = confusion_matrix(Ytest_BOW, y_pred_SVM_BOW)
# F1-score
f1_score(Ytest_BOW,y_pred_SVM_BOW)
#Predicting the Test set results
Test_pred_SVM_BOW = classifier_BOW_SVM.predict(Test_BOW)
dataset1['label'] = Test_pred_SVM_BOW
SVM_submission_BOW = dataset1[['id','label']]
SVM_submission_BOW.to_csv("SVM_submission_BOW_0.74.csv") 

#-----------------------------------------Tfidf Model-------------------------------------------------------
classifier_Tfidf_SVM = SVC(kernel = 'linear', random_state = 0)
classifier_Tfidf_SVM.fit(Xtrain_Tfidf, Ytrain_Tfidf)
y_pred_SVM_Tfidf = classifier_Tfidf_SVM.predict(Xtest_Tfidf)

#Confusion Matrix for error checking
cm_SVM_Tfidf = confusion_matrix(Ytest_Tfidf, y_pred_SVM_Tfidf)
# F1-score
f1_score(Ytest_Tfidf,y_pred_SVM_Tfidf)
#Predicting the Test set results
Test_pred_SVM_Tfidf = classifier_Tfidf_SVM.predict(Test_Tfidf)
dataset1['label'] = Test_pred_SVM_Tfidf
SVM_submission_Tfidf = dataset1[['id','label']]
SVM_submission_Tfidf.to_csv("SVM_submission_Tfidf_0.75.csv") 

#---------------------------------------------------Decision Tree---------------------------------------
#-------------------------------------------Bag of Words----------------------------------------------
from sklearn.tree import DecisionTreeClassifier
classifier_BOW_DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_BOW_DT.fit(Xtrain_BOW, Ytrain_BOW)

# Predicting the Test set results
y_pred_DT_BOW_ = classifier_BOW_DT.predict(Xtest_BOW)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

f1_score(Ytest_BOW,y_pred_DT_BOW_)














                 
                 
                 