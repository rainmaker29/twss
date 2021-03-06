# -*- coding: utf-8 -*-
"""twss_notebook.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17mlLa4xNC0N6UNRTEbTA3RXZyoA8azTK
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
# nltk.download()

# from google.colab import drive
# drive.mount('/content/drive')

fml = pd.read_csv('drive/My Drive/twss/fml.txt',sep="\n",header=None)
fml.columns=['text']
fml.head()

tfln = pd.read_csv('drive/My Drive/twss/tfln.onesent.txt',sep="\n",header=None,encoding='latin')
tfln.columns=['text']
tfln.head()

usa = pd.read_csv('drive/My Drive/twss/usaquotes.txt',sep="\n",header=None,error_bad_lines=False)
usa.columns = ['text']
usa.head()

twss = pd.read_csv('drive/My Drive/twss/twssstories.txt',sep="\n",header=None,encoding='latin')
twss.columns=['text']
twss.head()

print("fml : ",len(fml))
print("tfln : ",len(tfln))
print("usa : ",len(usa))
print("twss : ",len(twss))

twss['target'] = [1]*2027
twss.head()

usa = usa.loc[:666,:]
tfln = tfln.loc[:666,:]
fml = fml.loc[:666,:]

usa['target'] = [0]*667

tfln['target'] = [0]*667
fml['target'] = [0]*667

df = pd.concat([twss,usa,tfln,fml],ignore_index=True)

df

df = df.sample(frac=1).reset_index(drop=True)

df

y = df['target']
df.drop(['target'],inplace=True,axis=1)

xtrain,xvalid,ytrain,yvalid= train_test_split(df.text.values,y,stratify=y,random_state=42,test_size=0.1,shuffle=True)

print(xtrain.shape,xvalid.shape)

tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

tfv.fit(list(xtrain) + list(xvalid))
xtrain_tfv =  tfv.transform(xtrain)
xvalid_tfv = tfv.transform(xvalid)

clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)

print(metrics.accuracy_score(predictions,yvalid))

clf = MultinomialNB()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)

print(metrics.accuracy_score(predictions,yvalid))

svd = decomposition.TruncatedSVD(n_components=120)
svd.fit(xtrain_tfv)
xtrain_svd = svd.transform(xtrain_tfv)
xvalid_svd = svd.transform(xvalid_tfv)

# Scale the data obtained from SVD. Renaming variable to reuse without scaling.
scl = preprocessing.StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)
xvalid_svd_scl = scl.transform(xvalid_svd)

clf = SVC(C=1.0, probability=True) # since we need probabilities
clf.fit(xtrain_svd_scl, ytrain)
predictions = clf.predict(xvalid_svd_scl)

print(metrics.accuracy_score(predictions,yvalid))

import xgboost

xgb = xgboost.XGBClassifier(n_thread=10)
xgb.fit(xtrain_tfv,ytrain)

predictions = xgb.predict(xvalid_tfv)

print(metrics.accuracy_score(predictions,yvalid))

xgb.fit(xtrain_svd,ytrain)

predictions = xgb.predict(xvalid_svd)

print(metrics.accuracy_score(predictions,yvalid))

mll_scorer = metrics.make_scorer(metrics.accuracy_score, greater_is_better=True, needs_proba=False)



nb_model = MultinomialNB()

# Create the pipeline
clf = pipeline.Pipeline([('nb', nb_model)])

# parameter grid
param_grid = {'nb__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# Initialize Grid Search Model
model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

# Fit Grid Search Model
model.fit(xtrain_tfv, ytrain)  # we can use the full data here but im only using xtrain.
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

clf = MultinomialNB(alpha=0.1)
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)

print(metrics.accuracy_score(predictions,yvalid))

def tfconverter(x):
  tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
  tfv.fit(list(xtrain) + list(xvalid))

  return tfv.transform(x)

x = tfconverter([input()])
clf = MultinomialNB(alpha=0.1)
clf.fit(xtrain_tfv, ytrain)
res = clf.predict(x)

if res == np.array([0]):
  print("Nope")
else:
  print("That's what she said")

# clf

import pickle
pickle.dump(clf,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

x = tfconverter([input()])
model.predict(x)
