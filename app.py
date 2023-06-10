# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import pickle
import numpy as np
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
from flask import Flask, request, jsonify, render_template

app = Flask(__name__,template_folder='template')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    if output == 1:
         res_val = ' has diabetes ! The person is likely to have diabetes.'
    else:
         res_val ='is Normal ! The person is not likely to have diabetes.'
    return render_template('Diabetesresult.html', prediction_text = 'patient {}'.format(res_val)) 

if __name__ == '__main__':
    app.run(debug=True)
