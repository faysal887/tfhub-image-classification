from flask import Flask, render_template, request
import numpy as np
import base64
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from flask import Flask, request, jsonify
import io
import subprocess
from subprocess import Popen
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import PIL
import tensorflow as tf   
import json
from datetime import datetime


app = Flask(__name__)


models_dir='models/'
np.set_printoptions(suppress=True)


data_dir='./assets/'


def update_status_file(_dir, status='untracked'):
    df=pd.DataFrame()

    classes=os.listdir(_dir)

    for c in classes:
        mydir=f'{data_dir}/{c}/'
        onlyfiles = [f for f in listdir(mydir) if isfile(join(mydir, f))]
        print(c, len(onlyfiles))
        tmp=pd.DataFrame({'file_name': onlyfiles, 'class_name': [c]*len(onlyfiles), 'status': status})
        df=pd.concat([df, tmp])
        df.to_csv('train_status.csv', index=False)
    


@app.route('/classify', methods=['POST'])
def classify():
    try:
        
        df=pd.read_csv(f'{models_dir}models_hist.csv').reset_index(drop=True)
        selected=df[df.is_default==True]
        if not selected.empty:
            model_name=selected.model_name.item()
        else:
            model_name=df.iloc[df['accuracy'].argmax()].model_name
        
        
        model = tf.keras.models.load_model(f'{models_dir}{model_name}')
        
        img_bytes = request.files["image"]
        img = Image.open(io.BytesIO(img_bytes.read()))
        img = img.resize((224,224))
        img = np.array(img)
        img = img / 255.0
        img = img.reshape(1,224,224,3)
    
        y_pred=model.predict_classes(img)[0]
        predictions=model.predict(img)
        y_probs = np.array(tf.nn.sigmoid(predictions))[0]
        classes=os.listdir(data_dir)
        classes.sort()
        classes_probs=dict(zip(classes,y_probs))

        result={"y_pred":classes[y_pred], "probs": str(classes_probs)}
        
        return str(result)
        
    except Exception as e:
        return str(e)
        

@app.route('/add_train_img', methods=['POST'])
def add_train_img():
    try:
        file = request.files["image"]


        class_name = file.filename.split('.')[0]
        
        # dt=datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        dt=str(time.time()).replace('.','')
        
        file_name=f'{dt}_{class_name}'
        
        img = Image.open(io.BytesIO(file.read()))
        img = np.array(img)
        im = PIL.Image.fromarray(img)
        im.save(f"{data_dir}{class_name}/{file_name}")

        return "success"
    except Exception as e:
        return "error"


@app.route('/get_train_status', methods=['GET'])
def train_status():
    df=pd.read_csv('train_status.csv')

    status=df.groupby(['class_name', 'status']).size().to_dict()
    
    final=dict()
    for k,v in status.items():
        if k[0] not in final:
            final[k[0]]={k[1]: v}
        else:
            final[k[0]][k[1]] = v 
    print(final)
    return final


@app.route('/train', methods=['GET'])
def train():
    try:
        update_status_file(data_dir, status='tracked')
        epochs=request.args.get("epochs")
        p=subprocess.Popen(["python", "train.py", epochs])
        return "success!!!"       
    except:
        return "error"
    

@app.route('/select_model', methods=['GET'])
def select_model():
    model_name=request.args.get("model_name")
    
    df=pd.read_csv(f'{models_dir}models_hist.csv')

    df.loc[:, 'is_default']=False
    df.loc[df.model_name==model_name, 'is_default']=True
    
    df.to_csv(f'{models_dir}models_hist.csv', index=False)

    return "success"


    
@app.route('/list_models', methods=['GET'])
def list_models():

    df=pd.read_csv(f'{models_dir}models_hist.csv')
    df.set_index('model_name', inplace=True)
    models_dict = df.to_dict(orient='index')
    
    return jsonify(models_dict)


        
if __name__ == '__main__':
#     update_status_file(data_dir)
    app.run(debug=True, port=5000)
    
    
    
# curl -X POST -F image=@test_photo.jpg 'http://localhost:5000/classify'













