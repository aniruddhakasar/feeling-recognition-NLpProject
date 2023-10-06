from flask import Flask,render_template,jsonify,request
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')
model = load_model('generalized_model.h5')
ps = PorterStemmer()
app=Flask(__name__)
@app.route('/',methods=['GET','POST'])
def Home():
    
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method=='POST':
        name = request.form['name']
        text = request.form['text']
        data = [text]
        cleaned = data[0].replace('\n',' ')
        data[0] = cleaned
        data = re.sub('[^a-zA-Z]',' ',data[0])
        ls_of_word = nltk.word_tokenize(data)
        removed_stopwords = [ps.stem(word) for word in ls_of_word if word not in set(stopwords.words('english'))]
        cleaned_data = ' '.join(removed_stopwords)
        stemed_data = []
        stemed_data.append(cleaned_data)
        vocabulary_size = 20000
        one_hoted = [one_hot(stemed_data[0],vocabulary_size) ]
        embedded_data  = pad_sequences(one_hoted,padding='pre',maxlen=208)
        x_test = np.array(embedded_data)
        y_pred = model.predict(x_test)
        pred_index = [np.argmax(element) for element in y_pred]
        index_for_label = int(pred_index[0])
        label = ['joy','sadness','anger','fear','Love']
        return render_template('index.html',prediction=label[index_for_label])
if __name__ == "__main__":
    app.run(debug=True)