import pandas as pd
import flask
import numpy as np
import pickle


app = flask.Flask(__name__, template_folder='templates')

model = pickle.load(open('model/best_model_logreg.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

@app.route('/')
def main():
    return flask.render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)

@app.route("/predict", methods=['GET','POST'])
def predict():
    if flask.request.method=='POST':
        Rainfall = flask.request.form['Rainfall']
        Humidity9am = flask.request.form['Humidity9am']
        Humidity3pm = flask.request.form['Humidity3pm']
        Cloud9am = flask.request.form['Cloud9am']
        Cloud3pm = flask.request.form['Cloud3pm']
        RainToday = flask.request.form['RainToday']
        Sunshine = flask.request.form['Sunshine']
        WindGustSpeed = flask.request.form['WindGustSpeed']
        Pressure9am = flask.request.form['Pressure9am']
        Pressure3pm = flask.request.form['Pressure3pm']

        predict_list = [RainToday, Cloud9am, Cloud3pm, Humidity9am, Humidity3pm, Rainfall, Sunshine, WindGustSpeed, Pressure9am, Pressure3pm]
        sample = np.array(predict_list, dtype=float).reshape(1,-1)
        sample_scaled = scaler.transform(sample)
        prediction = model.predict(sample_scaled)
        
        output = {0: 'tidak akan hujan.', 1: 'akan hujan!'}
        
        if prediction == 0:
            return flask.render_template('predict1.html', prediction_text='Besok {}'.format(output[prediction[0]]))
        elif prediction == 1:
            return flask.render_template('predict.html', prediction_text='Besok {}'.format(output[prediction[0]]))
        else:
            return flask.render_template('home.html')
    else:
        return flask.render_template('home.html')