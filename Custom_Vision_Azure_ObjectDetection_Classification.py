# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 17:58:31 2018

@author: mayank.khandelwal
"""

'''
This code shows how the Azure Custom Vision service can be consumed for Classification API as well as Object Detection API
Results Displayed Almost Realtime using CV2 (OpenCV)
There are scopes for improvements and this code is to be used for basic reference only.
'''

import os
import sys
import numpy as np
import cv2
import http.client, urllib.request, urllib.parse, urllib.error, base64, json
import pandas as pd
import threading

conn_obj = http.client.HTTPSConnection('southcentralus.api.cognitive.microsoft.com')
conn_class = http.client.HTTPSConnection('southcentralus.api.cognitive.microsoft.com')
answer=""
image_frame = None
boundary = None

def get_prediction_classification():
    global answer
    headers = {
            'Prediction-Key': 'xxxxxxxxxxxxx',
            'Content-Type': 'multipart/form-data'}

    params = urllib.parse.urlencode({
            'iterationId': '{string}',
            'application': '{string}',})

    conn_class.request("POST", "/customvision/v2.0/Prediction/xxxxxxxxxxxxx/image?%s" % params, image_frame, headers)
    response = conn_class.getresponse()
    data = pd.DataFrame.from_dict(json.loads((response.read()).decode("utf-8"))['predictions'])
    answer = (data.iloc[data['probability'].idxmax(),2])
    conn_class.close()
    print(answer)

def get_prediction_obj_detection():
    global answer
    global boundary
    headers = {
            'Prediction-Key': 'xxxxxxxxxxxxx',
            'Content-Type': 'multipart/form-data'}
    
    params = urllib.parse.urlencode({
            'iterationId': '{string}',
            'application': '{string}',})
    
    conn_obj.request("POST", "/customvision/v2.0/Prediction/xxxxxxxxxxxxx/image?%s" % params, image_frame, headers)
    response = conn_obj.getresponse()
    data = pd.DataFrame.from_dict(json.loads((response.read()).decode("utf-8"))['predictions'])
    boundary = data.boundingBox[0]
    conn_obj.close()

video = cv2.VideoCapture(1)
ret = video.set(3,1280)
ret = video.set(4,720) 
width = video.get(3)
height = video.get(4)
frame = None
result_obj = threading.Thread(target=get_prediction_obj_detection)
result_class = threading.Thread(target=get_prediction_classification)
while(True):    
    ret, frame = video.read()  
    content = cv2.imencode('.jpg', frame)[1].tostring()
    
    try:
        cv2.rectangle(frame,
                      (int(width*boundary["left"]),int(height*boundary["top"])),
                      (int(width*boundary["width"]),int(height*boundary["height"])),
                      (0,255,0),3)
    except:
        pass
    cv2.imshow('Object detector', frame)
    
    if cv2.waitKey(1) == ord('z'):
        if not result_obj.isAlive():
            image_frame = content
            result_class = threading.Thread(target=get_prediction_classification)
            rqqesult_class.start()
            result_obj = threading.Thread(target=get_prediction_obj_detection)
            result_obj.start()

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()