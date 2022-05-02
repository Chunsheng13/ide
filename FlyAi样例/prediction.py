# -*- coding: utf-8 -*
import os
import numpy as np
import cv2
from flyai.framework import FlyAI
from keras.models import load_model
from path import MODEL_PATH, DATA_PATH
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# tensorflow2.1 keras2.3.1需要加入下面三行，否则在服务器上运行会报错
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class_num = ['Danaus_chrysippus', 'Losaria_coon',......]
class Prediction(FlyAI):
    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        model_path = os.path.join(MODEL_PATH, 'model.h5')
        self.model = load_model(model_path)
    def predict(self, image_path):
        '''
        模型预测返回结果
        :param input: 评估传入样例 {"image_path": "./data/input/image/172691.jpg"}
        :return: 模型预测成功中,返回预测结果格式 {"label": "Losaria_coon"}
        # 假设图片image_path对应的标签为Losaria_coon， 则返回格式为 {"label": "Losaria_coon"}
        '''
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = np.array([image]) / 255.
        preds = self.model.predict(image)[0]
        pred = np.argmax(preds)
        return {"label": class_num[int(pred)]}