# 这是一个示例 Python 脚本。
'''
构建自己的模型:我们可以在Main类之外，编写模型代码，也单独可以建立一个模型文件，如net.py，然后调用即可。
将训练好的模型保存到MODEL_PATH路径下，MODEL_PATH的定义可以查看path.py文件。
此为分类任务
'''
# -*- coding: utf-8 -*-
import argparse
import os
import cv2
from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
from path import MODEL_PATH
from net import mymodel
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# tensorflow2.1 keras2.3.1需要加入下面三行，否则在服务器上运行会报错
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()
class_dict = {'Danaus_chrysippus': 0, 'Losaria_coon': 1,......} # 这里略写了，需要补充
class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''
    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids("ButterflyClassification")
    def deal_with_data(self):
        '''
        处理数据，没有可不写。
        :return:
        '''
        df = pd.read_csv(os.path.join(DATA_PATH, 'ButterflyClassification', 'train.csv'))
        image_path_list = df['image_path'].values
        label_list = df['label'].values
        x_data = []
        y_data = []
        for image_path, label in zip(image_path_list, label_list):
            image_path = os.path.join(DATA_PATH, 'ButterflyClassification', image_path)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            x_data.append(image)
            y_data.append(class_dict[label])
        self.x_data = np.array(x_data) / 255.
        self.y_data = np.array(y_data)
    def train(self):
        '''
        训练模型，必须实现此方法
        :return:
        '''
        model = mymodel()  # 这个model需要自己编写
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
        model_path = os.path.join(MODEL_PATH, 'model.h5')
        mp = ModelCheckpoint(filepath=model_path, save_best_only=True, save_weights_only=False, mode='min', monitor='val_loss', verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='auto', factor=0.1, patience=3, verbose=1)
        el = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
        cllist = [mp, reduce_lr, el]
        batch_size = 16
        his = finalmodel.fit(self.x_data, self.y_data,
                    batch_size=batch_size,
                    verbose=2,
                    epochs=20,
                    validation_split=0.1,
                    callbacks=cllist,
                    )
if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.train()
