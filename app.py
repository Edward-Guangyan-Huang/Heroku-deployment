import numpy as np
from flask import Flask, request,render_template
import pickle
from transitioner import Transitioner

app = Flask(__name__)
logmodel = pickle.load(open('logmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    #final_features = [np.array(features)]
    prediction = logmodel.predict([Transitioner(features)])

    output = prediction[0]

    return render_template('index2.html', prediction_text='Your choice for president should be {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
