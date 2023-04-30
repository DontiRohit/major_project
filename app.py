from flask import Flask, render_template, request, send_file
from PIL import Image

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2 as cv
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input,decode_predictions
from keras import backend as K
from keras.layers import add, Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization, RepeatVector, Reshape
from keras.layers import concatenate
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

tf.random.set_seed(2)
np.random.seed(1)
# print(os.listdir("../input"))

InputPath="./download.jpeg"

# def noisy(noise_typ,image):
#     if noise_typ == "gauss":
#         row,col,ch= image.shape
#         mean = 0
#         var = 0.0001
#         sigma = var**0.05
#         gauss = np.random.normal(mean,sigma,(row,col,ch))
#         gauss = gauss.reshape(row,col,ch)
#         noisy =  gauss + image
#         return noisy
#     elif noise_typ == "s&p":
#         row, col, ch = image.shape
#         print(row, col)  # add this line to check the size of the input image

#         row,col,ch = image.shape
#         s_vs_p = 0.5
#         amount = 1.0
#         out = np.copy(image)
#         # Salt mode
#         num_salt = np.ceil(image.size * s_vs_p)
#         coords = [np.random.randint(0, i, int(num_salt))
#               for i in image.shape]
#         out[coords] = 1

#         # Pepper mode
#         num_pepper = np.ceil(image.size * (1. - s_vs_p))
#         coords = [np.random.randint(0, i , int(num_pepper))
#               for i in image.shape]
#         out[coords] = 1
#         return out
#     elif noise_typ == "poisson":
#         vals = len(np.unique(image))
#         vals = 2 ** np.ceil(np.log2(vals))
#         noisy = np.random.poisson(image * vals) / float(vals)
#         return noisy
#     elif noise_typ =="speckle":
#         row,col,ch = image.shape
#         gauss = np.random.randn(row,col,ch)
#         gauss = gauss.reshape(row,col,ch)        
#         noisy = image + image * gauss
#         return noisy

# img = cv.imread(InputPath)  
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# l = img.max()
# plt.imshow(img)

# img = Image.open(InputPath)
# img.show()


# Noise = noisy("s&p",img)
# plt.imshow(Noise)

# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) #convert it to hsv
# hsv[...,2] = hsv[...,2]*0.2
# img1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
# Noise2 = noisy("s&p",img)

# plt.imshow(Noise2)

# def PreProcessData(ImagePath):
#     X_=[]
#     y_=[]
#     count=0
#     for imageDir in os.listdir(ImagePath):
#         if count<2131:
#             try:
#                 count=count+1
#                 img = cv.imread(ImagePath + imageDir)
#                 img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#                 img_y = cv.resize(img,(500,500))
#                 hsv = cv.cvtColor(img_y, cv.COLOR_BGR2HSV) #convert it to hsv
#                 hsv[...,2] = hsv[...,2]*0.2
#                 img_1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
#                 Noisey_img = noisy("s&p",img_1)
#                 X_.append(Noisey_img)
#                 y_.append(img_y)
#             except:
#                 pass
#     X_ = np.array(X_)
#     y_ = np.array(y_)
    
#     return X_,y_
# print(os.getcwd())
# X_,y_ = PreProcessData("./LOLdataset/eval15/high/")

K.clear_session()
def InstantiateModel(in_):
    
    model_1 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(in_)
    model_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_1)
    model_1 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_1)
    
    model_2 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(in_)
    model_2 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_2)
    
    model_2_0 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_2)
    
    model_add = add([model_1,model_2,model_2_0])
    
    model_3 = Conv2D(64,(3,3), activation='relu',padding='same',strides=1)(model_add)
    model_3 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_3)
    model_3 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_3)
    
    model_3_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_add)
    model_3_1 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_3_1)
    
    model_3_2 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_add)
    
    model_add_2 = add([model_3_1,model_3_2,model_3])
    
    model_4 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add_2)
    model_4_1 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add)
    #Extension
    model_add_3 = add([model_4_1,model_add_2,model_4])
    
    model_5 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add_3)
    model_5 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_add_3)
    
    model_5 = Conv2D(3,(3,3), activation='relu',padding='same',strides=1)(model_5)
    
    return model_5

Input_Sample = Input(shape=(500, 500,3))
Output_ = InstantiateModel(Input_Sample)
Model_Enhancer = Model(inputs=Input_Sample, outputs=Output_)

Model_Enhancer.compile(optimizer="adam", loss='mean_squared_error')
# Model_Enhancer.summary()

# from keras.utils.vis_utils import plot_model
# plot_model(Model_Enhancer,to_file='model_.png',show_shapes=True, show_layer_names=True)
# from IPython.display import Image
# Image(retina=True, filename='model_.png')

# def GenerateInputs(X,y):
#     for i in range(len(X)):
#         X_input = X[i].reshape(1,500,500,3)
#         y_input = y[i].reshape(1,500,500,3)
#         yield (X_input,y_input)
# Model_Enhancer.fit(GenerateInputs(X_,y_),epochs=53,verbose=1,steps_per_epoch=39,shuffle=True)

import warnings
import joblib
warnings.filterwarnings("ignore")

print(os.getcwd())
joblib.dump(InstantiateModel(Input_Sample),"job")

TestPath="./download.jpeg"

# def ExtractTestInput(ImagePath):
#     img = cv.imread(ImagePath)
#     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#     img_ = cv.resize(img,(500,500))
#     hsv = cv.cvtColor(img_, cv.COLOR_BGR2HSV) #convert it to hsv
#     hsv[...,2] = hsv[...,2]*0.2
#     img1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
#     Noise = noisy("s&p",img1)
#     Noise = Noise.reshape(1,500,500,3)
#     return Noise

# ImagePath=InputPath
# image_for_test = ExtractTestInput(ImagePath)
# Prediction = Model_Enhancer.predict(image_for_test)
# Prediction = Prediction.reshape(500,500,3)
# TestPath2="./2.jpg"

# Image_test2=TestPath2
# plt.figure(figsize=(30,30))
# # plt.subplot(5,5,1)
# img_1 = cv.imread(Image_test2)
# img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)
# img_1 = cv.resize(img_1, (500, 500))
# plt.title("Ground Truth",fontsize=20)
# # plt.imshow(img_1)

# plt.subplot(5,5,1+1)
# # img_ = ExtractTestInput(Image_test2)
# Prediction = Model_Enhancer.predict(img_)
# img_ = img_.reshape(500,500,3)
# plt.title("Low Light Image",fontsize=20)
# plt.imshow(img_)

# plt.subplot(5,5,1+2)
# Prediction = Prediction.reshape(500,500,3)
# img_[:,:,:] = Prediction[:,:,:]
# plt.title("Enhanced Image",fontsize=20)
# plt.imshow(img_)

# Image_test2="./1.jpg"
# plt.figure(figsize=(30,30))
# # plt.subplot(5,5,1)
# img_1 = cv.imread(Image_test2)
# img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)
# img_1 = cv.resize(img_1, (500, 500))
# plt.title("Ground Truth",fontsize=20)
# # plt.imshow(img_1)

# plt.subplot(5,5,1+1)
# # img_ = ExtractTestInput(Image_test2)
# Prediction = Model_Enhancer.predict(img_)
# img_ = img_.reshape(500,500,3)
# plt.title("Low Light Image",fontsize=20)
# plt.imshow(img_)

# plt.subplot(5,5,1+2)
# Prediction = Prediction.reshape(500,500,3)
# img_[:,:,:] = Prediction[:,:,:]
# plt.title("Enhanced Image",fontsize=20)
# plt.imshow(img_)

# Image_test2="./3.jpeg"
# plt.figure(figsize=(30,30))
# # plt.subplot(5,5,1)
# img_1 = cv.imread(Image_test2)
# img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)
# img_1 = cv.resize(img_1, (500, 500))
# plt.title("Ground Truth",fontsize=20)
# # plt.imshow(img_1)

# plt.subplot(5,5,1+1)
# # img_ = ExtractTestInput(Image_test2)
# Prediction = Model_Enhancer.predict(img_)
# img_ = img_.reshape(500,500,3)
# plt.title("Low Light Image",fontsize=20)
# plt.imshow(img_)

# plt.subplot(5,5,1+2)
# Prediction = Prediction.reshape(500,500,3)
# img_[:,:,:] = Prediction[:,:,:]
# plt.title("Enhanced Image",fontsize=20)
# plt.imshow(img_)

# test_image="./3.jpeg"
# img = cv.imread(test_image)
# # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# img_ = cv.resize(img,(500,500))
# # hsv = cv.cvtColor(img_, cv.COLOR_BGR2HSV) #convert it to hsv
# # hsv[...,2] = hsv[...,2]*0.2
# # img1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
# # Noise = noisy("s&p",img1)
# img_ =img_.reshape(1,500,500,3)

# model=joblib.load("job")
# plt.figure(figsize=(30,30))
# # plt.subplot(5,5,1+1)
# # img_ = ExtractTestInput(test_image)
# Prediction = Model_Enhancer.predict(img_)
# img_ = img_.reshape(500,500,3)
# plt.title("Low Light Image",fontsize=20)
# plt.imshow(img_)

# plt.subplot(5,5,1+2)
# Prediction = Prediction.reshape(500,500,3)
# img_[:,:,:] = Prediction[:,:,:]
# plt.title("Enhanced Image",fontsize=20)
# plt.imshow(img_)

# test_image="./146.png"
# img = cv.imread(test_image)
# # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# img_ = cv.resize(img,(500,500))
# # hsv = cv.cvtColor(img_, cv.COLOR_BGR2HSV) #convert it to hsv
# # hsv[...,2] = hsv[...,2]*0.2
# # img1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
# # Noise = noisy("s&p",img1)
# img_ =img_.reshape(1,500,500,3)

# model=joblib.load("job")
# plt.figure(figsize=(20,10))
# # plt.subplot(5,5,1+1)
# # img_ = ExtractTestInput(test_image)
# Prediction = Model_Enhancer.predict(img_)
# img_ = img_.reshape(500,500,3)
# plt.title("Low Light Image",fontsize=20)
# plt.imshow(img_)



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/enhance', methods=['POST'])
def enhance():
    # Open the uploaded image and apply image enhancement techniques
    image = Image.open(request.files['myImage'])
    # enhanced_image = image.enhance()
    gt_image_file="gt_image.jpg"
    
    image.save("C:\\Users\\User\\Downloads\\KMIT_MAJOR-main\\KMIT_MAJOR-main\\front_end\\flask\\static\\images\\"+gt_image_file)

    test_image="C:\\Users\\User\\Downloads\\KMIT_MAJOR-main\\KMIT_MAJOR-main\\front_end\\flask\\static\\images\\"+gt_image_file
    # plt.figure(figsize=(30,30))
    # plt.subplot(5,5,1)
    img_1 = cv.imread(test_image)
    img_ = cv.resize(img_1, (500, 500))
    img_ =img_.reshape(1,500,500,3)
    # plt.title("Ground Truth",fontsize=20)
    # plt.imshow(img_1)

    # plt.subplot(5,5,1+1)

    Prediction = Model_Enhancer.predict(img_)
    img_ = img_.reshape(500,500,3)
    plt.title("Low Light Image",fontsize=20)
    plt.imshow(img_)
    a=np.array(img_)
    a1=Image.fromarray(a)
    low_image_file="low_image.jpg"
    a1.save("C:\\Users\\User\\Downloads\\KMIT_MAJOR-main\\KMIT_MAJOR-main\\front_end\\flask\\static\\images\\"+low_image_file)

    
    Prediction = Prediction.reshape(500,500,3)
    img_[:,:,:] = Prediction[:,:,:]
   
    
    
    img = np.array(img_)
    enhanced_image_file = "enhanced_image.jpg"
    final_img = Image.fromarray(img)
    final_img.save("C:\\Users\\User\\Downloads\\KMIT_MAJOR-main\\KMIT_MAJOR-main\\front_end\\flask\\static\\images\\"+enhanced_image_file)
    image_path = "/static/images/"
    result=True
    return render_template('index.html', result=result, original_image= image_path+low_image_file, enhanced_image= image_path+gt_image_file)


@app.route('/download')
def download():
    # Send the enhanced image file to the user
    return send_file("enhanced_image.jpg", as_attachment=True)

if __name__ == '__main__':
    app.run()