from __future__ import print_function, division

from flask import Flask, request, current_app
from flask.json import JSONEncoder
from flask_cors import CORS
from sqlalchemy import create_engine, text
import json
import cv2
import numpy as np
import urllib.request
import dbModule
from PIL import Image
import base64
from io import BytesIO


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
plt.ion()   # interactive mode

app = Flask(__name__)
cors = CORS(app)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model_ft = models.resnet18(pretrained=True)
model_ft = torch.load("mobilenetv3_model_conv_crack.pth")
#model_ft = torch.load("/content/drive/MyDrive/Colab Notebooks/소융대 학술제/model_ft.pth")

# num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

def transform_image(image):
    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image = data_transforms['val'](image).unsqueeze(0)                                  # PyTorch 모델은 배치 입력을 예상하므로 1짜리 배치를 만듦
    return image

def get_prediction(image):
    outputs = model_ft(image)
    _, preds = torch.max(outputs, 1)

    print('outputs:', outputs)

    #print('preds', preds)
    class_names = ['no','yes']
    print('predicted: {}'.format(class_names[preds]))

    return class_names[preds]

#-----------------------------------------------------------------------------------------------------------------
def img_to_mosaic(url):

    # url to image
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    src = image
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(src_gray, 1.3, 5)
    ratio = 0.1

    for x, y, w, h in faces:
        small = cv2.resize(src[y: y + h, x: x + w], None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
        src[y: y + h, x: x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    img_file = Image.fromarray(src) # array -> image

    buffered = BytesIO()
    img_file.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()) # image -> url
    img_base64 = bytes("data:image/png;base64,", encoding='utf-8') + img_str

    return img_file, img_base64

@app.route('/')
def hello():
    return 'Hello, My First Flask!'



@app.route('/<string:id>/locimgsubmit', methods=['POST'])
def loc_imgsubmit(id):
    print('here')
    params = request.get_json()
    # params = json.loads(request.get_data(), encoding='utf-8')
    # if len(params) == 0:
    #     return 'No parameter'

    # params_str = ''
    # for key in params.keys():
    #     params_str += 'key: {}, value: {}<br>'.format(key, params[key]) #url 받고
    # print(params_str)
    img = params['img']

    # 모자이크--------------------------------------------------------------
    file, src = img_to_mosaic(img)
    print(src)

    # resp = urllib.request.urlopen(src)
    # image = np.asarray(bytearray(resp.read()), dtype='uint8')
    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # cv2.imshow(image)

    # DB에 모자이크 파일 넣기----------------------------------------------------------------------------------

    print(len(src), id)
    db_class = dbModule.Database()
    src = str(src)
    src = src[2:]
    print(src)
    # sql = "UPDATE LC_call_cam \
    #         SET cam_img = %s \
    #         WHERE userID = %s", (src, id)
    db_class.execute("UPDATE LC_call_cam SET cam_img = %s WHERE userID = %s", (src, id,))
    db_class.commit()
    
    # 수해인식 모델-------------------------------------------------------------------------------------------
    
    if file is not None:
        input_tensor = transform_image(file)
        prediction_idx = get_prediction(input_tensor)
        print(prediction_idx)

    
    return img

if __name__ == '__main__':
    app.run()