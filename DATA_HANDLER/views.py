from django.http import JsonResponse
from django.shortcuts import render
from django.http import JsonResponse
import boto3
import pandas as pd
import subprocess
import base64
import time
import json
import numpy as np
import cv2
import os
from random import randint
from fer import FER
import matplotlib.pyplot as plt
import pickle

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

EEG_GSR_MODEL_PATH = 'model.pkl'
EEG_GSR_DATA_PATH = 'data.csv'
GLOBAL_ROOM_TEMPERATURE = None


# LOW_STRESS_EEG = 

def get_randomized_eeg():
    df = pd.read_csv(EEG_GSR_DATA_PATH)
    random_index = randint(0, 252)
    row = df.iloc[random_index].drop(['EMOTION_1', 'EMOTION_2', 'GSR_4', 'EMOTION']).to_numpy()
    # print(row)
    return list(row)[1:]

# GLOBAL_EEG = get_randomized_eeg()
GLOBAL_IMAGE_EMOTION = None

# replace the key with the groups private key
privateKeyPath = os.path.join(os.getcwd(),'static')
privateKeyPath = os.path.join(privateKeyPath,'privateKey.json')
cred_obj = credentials.Certificate(privateKeyPath)

default_app = firebase_admin.initialize_app(cred_obj)
db = firestore.client()

s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-2',
    aws_access_key_id='AKIA3H3DVCTSIEELEHZR',
    aws_secret_access_key='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'
)

s3_client = boto3.client(
    service_name='s3',
    region_name='us-east-2',
    aws_access_key_id='AKIA3H3DVCTSIEELEHZR',
    aws_secret_access_key='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'
)

def save_img(img):
    img_dir = "Images"
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    ts = time.time()
    imageName = f"{img_dir}/image_{ts}.jpeg"
    cv2.imwrite(imageName, img)
	# print("Image Saved", end="\n") # debug
    # s3.Bucket('moodbucket1').upload_file(Filename=imageName, Key=imageName,object_name="image-data/")
    s3_client.upload_file(imageName, 'moodbucket1', f'image-data/{imageName}')

# Create your views here.
def addSensorData(request):
    status_code = 200
    status_message = "ok"
    try:
        input_data = json.loads(request.body)
        print(input_data)
        #update data to firebase
        doc_ref = db.collection(u'sensor-data')
        doc = doc_ref.document(u'data').get()
        sensor_data = doc.to_dict()
        gsr_values = sensor_data['GSR_Values']
        gsr_values.append(input_data['GSR'])
        sensor_data['GSR_Values'] = gsr_values[-10:]
        # ypred, eeg = detect_stress_from_eeg_gsr(float(input_data['GSR']))
        sensor_data['EEG_Values'] = get_randomized_eeg()
        global GLOBAL_ROOM_TEMPERATURE
        if not GLOBAL_ROOM_TEMPERATURE:
            GLOBAL_ROOM_TEMPERATURE = input_data['temperature']
            sensor_data['room_temperature'] = str(input_data['temperature'])
        else:
            sensor_data['body_temperature'] = str(input_data['temperature'])
        doc_ref.document(u'data').set(sensor_data)
        print('Firebase data added', doc_ref)
        return JsonResponse({"status":status_message,"status_code":status_code})
    except Exception as e:
        return JsonResponse({"status":"failed","status_code":400})
    
def addImageData(request):
    status_code = 200
    status_message = "ok"
    received = request
    if received.FILES:
        # convert string of image data to uint8
        file  = received.FILES['imageFile']
        nparr = np.fromstring(file.read(), np.uint8)
		# decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        emotion = detect_stress(img)
        global GLOBAL_IMAGE_EMOTION
        GLOBAL_IMAGE_EMOTION = emotion
        
    else:
        print(type(request))
        print("Nothing like that")

    return JsonResponse({"status":status_message,"status_code":status_code,"emotion":emotion})

def uploadToS3(data): 
    # Load the existing file from s3
    obj = s3.Bucket('moodbucket1').Object('SensorData.csv').get()
    dataFile = pd.read_csv(obj['Body'], index_col=0)

    #Append with the input data
    result = pd.concat([dataFile,data])
    result.to_csv('SensorData.csv')

    # Upload files to S3 bucket
    s3.Bucket('moodbucket1').upload_file(Filename='SensorData.csv', Key='SensorData.csv')

def play_audio(request):
    inputFilePath = 'alpha_music.wav'
    command = "open %s" % (inputFilePath)
    subprocess.call(command, shell=True)
    return JsonResponse({"Status":"Playing Music as BioFeedback, Keep Calm!"})

# def get_most_recent_s3_object(bucket_name, prefix):
#     s3 = boto3.client('s3')
#     paginator = s3.get_paginator( "list_objects_v2" )
#     page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
#     latest = None
#     for page in page_iterator:
#         if "Contents" in page:
#             latest2 = max(page['Contents'], key=lambda x: x['LastModified'])
#             if latest is None or latest2['LastModified'] > latest['LastModified']:
#                 latest = latest2
#     return latest

    
def detect_stress(image):
    # img = get_most_recent_s3_object('moodbucket1', 'moodbucket1/image-data/')
    # image = plt.imread(img)
    emo_detector = FER(mtcnn=True)

    image = cv2.rotate(image, cv2.ROTATE_180)
    save_img(image)

    # Capture all the emotions on the image
    captured_emotions = emo_detector.detect_emotions(image)
    # Print all captured emotions with the image
    print(captured_emotions)
    #plt.imshow(image)
    dominant_emotion, emotion_score = emo_detector.top_emotion(image)
    result = {
        'status' : 200,
        'message': " emotion detected",
        'dominant_emotion':dominant_emotion,
        'emotion_score':emotion_score
    }
    print(result)
    return result
    # return None

def detect_stress_from_eeg_gsr(gsr_data):
    one_hot_encoded = {
        0: 'VeryLow',
        1: 'Low',
        2: 'Moderate',
        3: 'High',
        4: 'VeryHigh'
    }

    with open(EEG_GSR_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    eeg = get_randomized_eeg()
    gsr = [gsr_data]
    arr = np.array(eeg + gsr).reshape(-1, 10)
    ypred = model.predict(arr)
    print(ypred)
    return one_hot_encoded[ypred[0]]


def predict(request):
    doc_ref = db.collection(u'sensor-data')
    doc = doc_ref.document(u'data').get()
    sensor_data = doc.to_dict()
    gsr_value = sensor_data['GSR_Values'][-1]
    sensor_data['music'] = False
    eeg_gsr_stress = detect_stress_from_eeg_gsr(gsr_value)
    # if not GLOBAL_IMAGE_EMOTION:
    print(eeg_gsr_stress)
    # if GLOBAL_IMAGE_EMOTION == 'sad' or GLOBAL_IMAGE_EMOTION == 'surprise':
    if eeg_gsr_stress == 'Moderate' or eeg_gsr_stress == 'High' or eeg_gsr_stress == 'VeryHigh':
        sensor_data['music'] = True
    doc_ref.document(u'data').set(sensor_data)

    return JsonResponse({
        'status': 'Result predicted',
        'playing_music': sensor_data['music']
    })
