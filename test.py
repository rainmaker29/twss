import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

model = pickle.load(open('model.pkl','rb'))

def tfconverter(x):
  tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
  tfv.fit(list(xtrain) + list(xvalid))

  return tfv.transform(x)
