from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import time
import nltk 
import random
import pickle
import json
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open("intents.json", 'rb') as file:
    data = json.load(file)
    words = []
    labels = []
    docs_x = []
    docs_y = []
    
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)
    file.close()


model = load_model('tf_chatbot')

def bag_of_words(sentence, words):
    bag = [[0 for _ in range(len(words))]]

    s_words = nltk.word_tokenize(sentence)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[0][i] = 1
                
    bag = np.array(bag)
            
    return bag

def predict(sentence):
    results = model.predict([bag_of_words(sentence, words)])
    results_index = np.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    return random.choice(responses)

chats = [[["Hello!", time.strftime("%H:%M:%S", time.localtime())], ["Nice to have you here! How may I serve you?", time.strftime("%H:%M:%S", time.localtime())]]]

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html', chats = chats)

@app.route('/', methods=['POST'])
def submit():
    if request.method == 'POST':
        message = request.form['message']
        new_chat = []
        income = []
        output = []
        income.append(message)
        income.append(time.strftime("%H:%M:%S", time.localtime()))
        output.append(predict(message))
        output.append(time.strftime("%H:%M:%S", time.localtime()))
        new_chat.append(income)
        new_chat.append(output)
        chats.append(new_chat)
    return render_template('index.html', chats = chats)

@app.route('/home', methods=['POST'])
def clear(chats = chats):
    if request.method == 'POST':
        chats.clear() 
        chats.append([["Hello!", time.strftime("%H:%M:%S", time.localtime())], ["Nice to have you here! How may I serve you?", time.strftime("%H:%M:%S", time.localtime())]])
    return render_template('index.html', chats = chats)

if __name__ == '__main__':
	app.run(debug=True)