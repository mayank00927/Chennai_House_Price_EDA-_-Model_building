from flask import Flask,request,render_template
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            AREA=request.form.get('AREA'),
            INT_SQFT=int(request.form.get('INT_SQFT')),
            N_BEDROOM=float(request.form.get('N_BEDROOM')),
            N_BATHROOM=float(request.form.get('N_BATHROOM')),
            N_ROOM=float(request.form.get('N_ROOM')),
            SALE_COND=request.form.get('SALE_COND'),
            PARK_FACIL=request.form.get('PARK_FACIL'),
            BUILDTYPE=request.form.get('BUILDTYPE'),
            STREET=request.form.get('STREET'),
            PROP_AGE=request.form.get('PROP_AGE'),
            PRICE_PER_SQ_FT=float(request.form.get('PRICE_PER_SQ_FT')),
            UTILITY=request.form.get('UTILITY')       
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)        