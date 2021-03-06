# Author: ariasulg  2021/03/06
# Before using this code, please contact the author:ariasulg2020@gmail.com

import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
import os
import cv2
from keras import applications
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image  # for load data
import tensorflow as tf

# for GPU OOM
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# for log
import time


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

debug = True


# load data
def load_data():
    train_path = "C:\\Users\\admin\\test\\DataSet\\Train\\train_pic"
    test_path = "C:\\Users\\admin\\test\\DataSet\\Test\\test_pic"

    count = 0
    train_set = []
    test_set = []
    for filename in os.listdir(train_path):
        img = Image.open(os.path.join(train_path,filename))
        np_img = np.array(img)
        train_set.append(np_img)
    for filename in os.listdir(test_path):
        img = Image.open(os.path.join(test_path,filename))
        np_img = np.array(img)
        test_set.append(np_img)

    import csv
    import pandas as pd
    train_csv="C:\\Users\\admin\\test\\DataSet\\Train\\2017_target_train.csv"
    test_csv="C:\\Users\\admin\\test\\DataSet\\Test\\2017_target_test.csv"
    train_label=[]
    with open(train_csv, 'r') as f:
        reader = csv.reader(f)
        for i in reader:
            train_label.append(i[2])
    train_label_pd = pd.DataFrame(train_label)
    train_label_pd.columns = ['label']
    print(train_label_pd)
    # print(train_label_pd.head(5))
    test_label=[]
    with open(test_csv, 'r') as f:
        reader = csv.reader(f)
        for i in reader:
            # print(i[3])
            test_label.append(i[2])
    test_label_pd = pd.DataFrame(test_label)
    test_label_pd.columns = ['label']
    print(test_label_pd)
    # print(test_label_pd.head(5))
    return (train_set,train_label_pd),(test_set,test_label_pd)

(x_train, y_train_pd), (x_test, y_test_pd) = load_data()


y_train_pd['label'] = y_train_pd['label'].astype('float')
y_test_pd['label'] = y_test_pd['label'].astype('float')

X_train = [cv2.resize(i, (105, 130)) for i in x_train]
X_test = [cv2.resize(i, (105, 130)) for i in x_test]

x_train = np.array(X_train)
x_test = np.array(X_test)
print("***************DataSet******************",
      "\nx_train.shape:",x_train.shape,
      "\ny_train.shape:",y_train_pd.shape,
      "\nx_test.shape:",x_test.shape,
      "\ny_test.shape:",y_test_pd.shape)

batch_size = 4
epochs = 20  
data_augmentation = True  
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_transfer_learning')  # Set the model save directory
model_name = 'keras_fashion_transfer_learning_trained_model.h5'  # Set the model name

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Color normalization
x_train /= 255  
x_test /= 255  

print('----------------train&test_pd.head(5)----------------------')
print("y_train_pd.head(5)")
print(y_train_pd.head(5))
print("y_test_pd.head(5)")
print(y_test_pd.head(5))

# Training set normalization
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(y_train_pd)
print("min_max_scaler.data_max_:",min_max_scaler.data_max_)
print("min_max_scaler.data_min_:",min_max_scaler.data_min_)
y_train = min_max_scaler.transform(y_train_pd)[:, 0]  # 归一化


# Verification set normalization
min_max_scaler.fit(y_test_pd)
y_test = min_max_scaler.transform(y_test_pd)[:, 0]  # 归一化

if debug:
    print("len(y_train):",len(y_train))
    print("len(y_test):",len(y_test))
print("---------------Finish normalization. Begin VGG16 model----------------")

# Using model
if debug:
    print("x_train.shape :",x_train.shape)  # (60000, 48, 48, 3)
    print("x_train.shape[1:] :",x_train.shape[1:])  
base_model = applications.InceptionV3(include_top=False, weights='imagenet', input_shape=x_train.shape[1:])  

# path to the model weights files.
# top_model_weights_path = 'bottleneck_fc_model.h5'
model = Sequential()
print(base_model.output)
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('linear'))

# add the model on top of the convolutional base
model = Model(inputs=base_model.input, outputs=model(base_model.output))  # VGG16模型与自己构建的模型合并
 
for layer in model.layers[:15]:
    layer.trainable = False

# initiate RMSprop optimizer  
opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='mae',
              optimizer=opt,
              )

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,  # x_train: input data，图片信息    y_train: target data，价格信息
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:  # Using picture augmentation
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    print("x_train.shape[0]//batch_size:",x_train.shape[0]//batch_size)  
    print("x_train.shape[0]/batch_size:",x_train.shape[0]/batch_size) 
    print("x_train.shape:",x_train.shape)
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3),
        tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ]
    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train,  
                                     batch_size=batch_size),
                        epochs=epochs,
                        steps_per_epoch=x_train.shape[0]//batch_size,
                        validation_data=(x_test, y_test),
                        # validation_steps=y_train.shape[0]//batch_size,
                        workers=4,  # The maximum number of processes that need to be started when using process-based threads.
                        callbacks=my_callbacks
                       )

model.summary()

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

import matplotlib.pyplot as plt
# Plotting loss values for training & validation
plt.plot(history.history['loss'],'r')
plt.plot(history.history['val_loss'],'b')

# Model Evaluation
score = model.evaluate(x_test, y_test, verbose=0)
print("Score:",score)

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Prediction
y_new = model.predict(x_test)

# Inverse normalization
min_max_scaler.fit(y_test_pd)
print("min_max_scaler.data_max_:",min_max_scaler.data_max_)
print("min_max_scaler.data_min_:",min_max_scaler.data_min_)
y_pred_pd = pd.DataFrame({'pred_label': list(y_new)})
y_new = min_max_scaler.inverse_transform(y_pred_pd)

y_test_pd['pred_label'] = y_new
print(y_test_pd.head(50))
# Documenting results
curTime=time.strftime('%Y-%m%d-%H%M',time.localtime(time.time()))+".csv"
print(curTime)
y_test_pd.to_csv(curTime)