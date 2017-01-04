import argparse

import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, MaxPooling3D, ELU
from keras.models import Sequential, model_from_json
from sklearn.model_selection import train_test_split
import json



# This method is for displaying the image from a file for debugging purpose
def getImage(file1, debug=False, plot=False):
    if (debug):
        print(file1)
    img = cv2.imread(file1)
    correctedimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if (plot):
        plotImage(correctedimg, debug)

    return correctedimg


# This method is for displaying the image from a file for debugging purpose
def plotImage(image, debug=False, title=""):
    if (debug):
        print(image.shape)
        print(image)
    plt.imshow(image)
    plt.figtext(0, 0, title)
    plt.show()


# Simple normalization , dividing the data by 255 and setting the mean to 0
def normalizeImage(image):
    image = image / 255.
    image -= 0.5
    return image


def getnormalizeddata(filelocations,yoriginaldata, debug=False):
    xnormalized = []
    for filename in filelocations:
        image = getImage(filename)
        resize = resizeimage(image)
        normalizedimage = normalizeImage(resize)
        xnormalized.append(normalizedimage)
        if debug and len(xnormalized) % 100 == 0:
            print("Loaded " + str(len(xnormalized)) + "images")

    ydata = yoriginaldata
    return xnormalized, ydata


def resizeimage(image):
    cropped = image[32:135, :]
    resized = cv2.resize(cropped, (200, 66), interpolation=cv2.INTER_AREA)
    return resized


def flipimage(image):
    flippedimage = cv2.flip(image, 1)
    return flippedimage


def augmentdata(xnormalizeddata, ydata):
    datalen = len(ydata)
    i = 0
    while i < datalen:
        currentx = xnormalizeddata[i]
        flippedimage = flipimage(currentx)
        xnormalizeddata.append(flippedimage)
        ydata = ydata.append(pd.Series(ydata[i] * -1))
        i = i + 1
    return xnormalizeddata, ydata


def getFinalData(imagefiles,yoriginaldata, augment=False):
    xnormalized, ydata = getnormalizeddata(imagefiles,yoriginaldata, debug=True)
    if (augment):
        xfinal, yfinal = augmentdata(xnormalized, ydata)
    else:
        xfinal = xnormalized
        yfinal = ydata
    return np.asarray(xfinal), np.asarray(yfinal)


def getimagedata(line_data):
    lrc_i = np.random.randint(3)
    path_file = None
    shift_ang = None
    if (lrc_i == 0 ):
        path_file = line_data[0].values[0].strip()
        shift_ang = 0
    elif (lrc_i == 1):
        path_file = line_data[1].values[0].strip()
        shift_ang = .25
    elif (lrc_i == 2):
        path_file = line_data[2].values[0].strip()
        shift_ang = -.25
    y_steer = line_data[3].values[0] + shift_ang

    xdata = getImage(path_file)
    resize = resizeimage(xdata)
    shadowed = add_random_shadow(resize)

    normalizedimage = normalizeImage(shadowed)
    return normalizedimage,y_steer



def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image


def generate_data(data,batch_size=128,pr_threshold=0.5):
    while 1:
        batch_x_data = []
        batch_y_data = []
        for i in range(batch_size):
            xdata = None
            ydata = None
            keep_pr = 0
            while keep_pr==0:
                rand_i = np.random.randint(len(data))
                line_data = data.loc[[rand_i]]
                xdata,ydata = getimagedata(line_data)
                if abs(ydata) < .1:
                    pr_val = np.random.uniform()
                    if pr_val>pr_threshold:
                        keep_pr = 1
                else:
                    keep_pr = 1
            image = xdata
            y = ydata
            fliprand = np.random.randint(2)
            if fliprand == 1:
                image = flipimage(image)
                y = y * -1
            batch_x_data.append(image)
            batch_y_data.append(y)
        yield np.asarray(batch_x_data),np.asarray(batch_y_data)



def trainmodel(data, batch_size=128, nb_epoch=20,model=None,pr_threshold_val=0.5):
    checkpoint = ModelCheckpoint("model.h5", monitor='loss', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]
    if model is None:
        model = Sequential()
        model.add(Convolution2D(24, 5, 5,
                                border_mode='valid',subsample=(2,2),init="he_normal",input_shape=(66, 200, 3,)))
        model.add(ELU())

        model.add(Convolution2D(36, 5, 5,
                                border_mode='valid',subsample=(2,2),init="he_normal"))
        model.add(ELU())
        model.add(Convolution2D(48, 5, 5,
                                border_mode='valid',subsample=(2,2),init="he_normal"))
        model.add(ELU())

        model.add(Convolution2D(64, 3, 3,
                                border_mode='valid',subsample=(1,1),init="he_normal"))
        model.add(ELU())

        model.add(Convolution2D(64, 3, 3,
                                border_mode='valid',subsample=(1,1),init="he_normal"))
        model.add(ELU())
        model.add(Flatten())
        model.add(ELU())
        model.add(Dense(1164,init="he_normal"))
        model.add(ELU())
        model.add(Dense(100,init="he_normal"))
        model.add(ELU())
        model.add(Dense(50,init="he_normal"))
        model.add(ELU())
        model.add(Dense(10,init="he_normal"))
        model.add(ELU())
        model.add(Dense(1,init="he_normal"))

        model.compile(loss='mean_squared_error', optimizer='adam')
    else:
        print("Loaded model from file")

    print(model.summary())
    model.fit_generator(generate_data(data,pr_threshold=pr_threshold_val),batch_size,nb_epoch,callbacks=callbacks_list)

    return model



def save_model(model, filename="model.h5",model_arch="model.json"):
    model.save(filepath=filename)
    model_json = model.to_json()
    with open(model_arch, 'w') as f:
        json.dump(model_json, f)
    print("Saved model to filename="+filename + ", and model_arch="+model_arch)




if __name__ == '__main__':
    dataframe = pd.read_csv("udacity_data/data/driving_log.csv", delim_whitespace=False, header=None)
    #centerimages = (dataframe[:][0])
    #yoriginal = (dataframe[:][3])
    #xfinal, yfinal = getFinalData(centerimages,yoriginal)

    # xgen,ygen = next(generate_data(xfinal,yfinal))
    # print(ygen)

    # X_train, X_valid, y_train, y_valid = train_test_split(
    #     xfinal,
    #     yfinal,
    #     test_size=0.05,
    #     random_state=832289)

    model = None
    if os.path.isfile("model.json"):
        with open('model.json', 'r') as jfile:
            if jfile is not None:
                model = model_from_json(json.load(jfile))
                model.compile("adam", "mse")
                weights_file = "model.h5"
                model.load_weights(weights_file)
                print("Loading model from file")

    model = trainmodel(dataframe ,batch_size=1280*10,nb_epoch=10 ,model=model,pr_threshold_val=0.5)
    model = trainmodel(dataframe ,batch_size=1280*10,nb_epoch=10 ,model=model,pr_threshold_val=0.9)
    model = trainmodel(dataframe ,batch_size=1280*10,nb_epoch=10 ,model=model,pr_threshold_val=0.0)
    save_model(model)

