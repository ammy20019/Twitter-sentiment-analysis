import tensorflow as tf

from tensorflow.keras.models import load_model

model = load_model('text_classifier_model/1/')

import pickle

with open('tfidfmodel.pickle','rb') as file:
    tfidf = pickle.load(file)

from flask_ngrok import run_with_ngrok
from flask import Flask, request, render_template

app = Flask(__name__)

run_with_ngrok(app)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def text_classifier():
    request_data = request.get_json(force=True)
    text = request_data['sentence']
    print("printing the sentence")
    print(text)
    text_list=[]
    text_list.append(text)
    print(text_list)
    numeric_text = tfidf.transform(text_list).toarray()
    output = model.predict(numeric_text)[:,1]
    print("Printing prediction")
    print(output)
    sentiment="unknown"
    if output[0] > 0.5 :
      print("positive prediction")
      sentiment="postive"
    else:
      print("negative prediction")
      sentiment="negative"
    print("Printing sentiment")     
    print(sentiment)
    return "The sentiment is {}".format(sentiment)

app.run()

