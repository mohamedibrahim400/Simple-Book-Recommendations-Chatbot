import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

# Load the model and other necessary files
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f'Found in bag: {w}')
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(ints, intents_json):
    if not ints:
        return "I'm sorry, I didn't quite catch that. Could you rephrase?"
    
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            responses = i['responses']
            # Check if it's a nested list like Philosophy: [[{Book...}, {Book...}]]
            if isinstance(responses, list) and len(responses) == 1 and isinstance(responses[0], list):
                responses = responses[0]
            
            result = random.choice(responses)
            
            # If the result is a dictionary (a book object), format it as a string
            if isinstance(result, dict):
                book_name = result.get('Book', 'Unknown Title')
                feedback = result.get('Feedback', '')
                rate = result.get('Rate', 'N/A')
                
                response_str = f"I recommend '{book_name}' (Rating: {rate}). "
                if feedback:
                    # Truncate feedback if too long for a single chat bubble
                    if len(feedback) > 300:
                        feedback = feedback[:297] + "..."
                    response_str += f"\n\nAbout the book: {feedback}"
                return response_str
            
            return result
    return "I'm not sure how to respond to that."

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    return res

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get')
def get_bot_response():
    user_text = request.args.get('msg')
    return chatbot_response(user_text)

if __name__ == '__main__':
    app.run(debug=True)
