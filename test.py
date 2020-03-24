import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

model = pickle.load(open('model.pkl','rb'))

def tfconverter(x,xtrain,xvalid):
  tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
  tfv.fit(list(xtrain) + list(xvalid))

  return tfv.transform(x)

df = pd.read_csv('dataframe.csv')
y = pd.read_csv('target.csv')

xtrain,xvalid,ytrain,yvalid= train_test_split(df.text.values,y,stratify=y,random_state=42,test_size=0.1,shuffle=True)

text = str(sys.argv[1])

text = tfconverter([text],xtrain,xvalid)

res = model.predict(text)

if res == np.array([1]):
    print("That's what she said")
else:
    print("Nope")
