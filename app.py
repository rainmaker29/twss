import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))

df = pd.read_csv('processed_data/dataframe.csv')
y = pd.read_csv('processed_data/target.csv')

xtrain,xvalid,ytrain,yvalid= train_test_split(df.text.values,y,stratify=y,random_state=42,test_size=0.1,shuffle=True)



def tfconverter(x,xtrain,xvalid):
  tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
  tfv.fit(list(xtrain) + list(xvalid))

  return tfv.transform(x)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)



    text = [str(x) for x in request.form.values()]
    text = text[0]

    text = tfconverter([text],xtrain,xvalid)

    res = model.predict(text)

    if res == np.array([1]):
        output = "That's what she said"
    else:
        output = "Nope"

    return render_template('index.html', prediction_text='{}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    # prediction = model.predict([np.array(list(data.values()))])
    prediction = model.predict(tfconverter(list(data.values),xtrain,xvalid))
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
