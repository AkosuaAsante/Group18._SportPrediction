import numpy as np
from flask import Flask, request,render_template
import pickle
from sklearn.preprocessing import StandardScaler as scaler

app=Flask(__name__)

model=pickle.load(open('model/midsemmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict(): 

    input_values = [int(x) for x in request.form.values()]
    features=[np.array(input_values)]
    prediction=model.predict(features)
    output = round(prediction[0], 2)
    print(output)
    return render_template('index.html',prediction_text=output)
    

if __name__=="__main__":
    app.run()



