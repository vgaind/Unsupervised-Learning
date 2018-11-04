import numpy as np
import os
from random import shuffle
import pandas as pd
import scipy.interpolate as interpolate
from subprocess import call
from skimage.feature import hog
from skimage.color import rgb2gray
import subprocess

def cmd_exists(cmd):
    return subprocess.call("type " + cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0

def download_cifar():
    print("downloading cifar")
    url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    call(["wget", "-c", url])
    store_file='cifar-10-python.tar.gz'
    if os.path.isdir(os.path.join(os.getcwd(), 'Dataset')) is False:
        call(['mkdir','Dataset'])
    call(['mv',store_file,'Dataset'])
    cwd=os.getcwd()
    os.chdir('Dataset')
    call(['tar','-xvzf',store_file])
    os.chdir(cwd)

def download_mnist():
    print("downloading mnist")
    url_train='https://pjreddie.com/media/files/mnist_train.csv'
    url_test='https://pjreddie.com/media/files/mnist_test.csv'
    call(["wget", "-c", url_train])
    call(["wget", "-c", url_test])
    if os.path.isdir(os.path.join(os.getcwd(), 'Dataset')) is False:
        call(['mkdir','Dataset'])
    call(['mv', 'mnist_train.csv', 'Dataset/'])
    call(['mv', 'mnist_test.csv', 'Dataset/'])

def unpickle2(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def reshape_cifar_images(image_matrix):
    image_matrix_split=np.split(image_matrix,axis=0)

def Cifar_data_arrange(cifar_dict,convert_to_GS=1,select_labels=[1,3],feature_extract=False):
    keylist = list(cifar_dict.keys())
    cifar_image_matrix = cifar_dict[b'data']
    batch_size = cifar_image_matrix.shape[0]
    cifar_image_matrix_split = np.split(cifar_image_matrix, 3, axis=1)
    colorR = np.expand_dims(np.reshape(cifar_image_matrix_split[0], (batch_size, 32, 32)), axis=3)
    colorG = np.expand_dims(np.reshape(cifar_image_matrix_split[1], (batch_size, 32, 32)), axis=3)
    colorB = np.expand_dims(np.reshape(cifar_image_matrix_split[2], (batch_size, 32, 32)), axis=3)
    cifar_image_matrix = np.concatenate((colorR, colorG, colorB), axis=3)

    if convert_to_GS:
        cifar_image_matrix = np.squeeze(np.mean(cifar_image_matrix,axis=3))

    if feature_extract:
        cifar_image_matrix_orig=cifar_image_matrix
        tempmatrix=[]
        for i in np.arange(cifar_image_matrix.shape[0]):
            tempmatrix.append(hog(cifar_image_matrix[i,:],block_norm='L2-Hys'))
        cifar_image_matrix=np.array(tempmatrix)



    all_label_set = np.array(cifar_dict[b'labels'])
    for sl in np.arange(len(select_labels)):
        index_sl = np.squeeze(np.where(all_label_set == select_labels[sl]))
        if sl == 0:
            image_set = cifar_image_matrix[index_sl, :]
            label_set = all_label_set[index_sl]
        else:
            image_set = np.concatenate((image_set, cifar_image_matrix[index_sl, :]), axis=0)
            label_set = np.concatenate((label_set, all_label_set[index_sl]), axis=0)
    shuffle_index = np.arange(label_set.shape[0])
    shuffle(shuffle_index)
    image_set = image_set[shuffle_index, :]
    label_set = label_set[shuffle_index]
    return image_set,label_set

def Load_Cifar(filedir, convert_to_GS=1,test_display=1,select_labels=[1,3],feature_extract=False):
    filename_list=['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']

    for filename in filename_list:

        cifar_dict=unpickle(os.path.join(filedir,filename))
        cifar_image_matrix_curr,label_set_curr=Cifar_data_arrange(cifar_dict=cifar_dict, convert_to_GS=convert_to_GS, select_labels=select_labels,feature_extract=feature_extract)
        if filename == 'data_batch_1':
            train_cifar_image_matrix=cifar_image_matrix_curr
            train_label_set=label_set_curr
        elif filename == 'test_batch':
            test_cifar_image_matrix=cifar_image_matrix_curr
            test_label_set=label_set_curr
        else:
            train_cifar_image_matrix=np.concatenate((train_cifar_image_matrix,cifar_image_matrix_curr),axis=0)
            train_label_set=np.concatenate((train_label_set,label_set_curr),axis=0)

    batch_size_train=train_cifar_image_matrix.shape[0]
    train_cifar_image_matrix=np.reshape(train_cifar_image_matrix,[batch_size_train,-1])
    batch_size_test = test_cifar_image_matrix.shape[0]
    test_cifar_image_matrix = np.reshape(test_cifar_image_matrix, [batch_size_test, -1])
    return train_cifar_image_matrix,train_label_set,test_cifar_image_matrix,test_label_set


def Load_Mnist(Data_Dir,Mnist_train_data_filename,Mnist_test_data_filename,feature_select=True):

    df_mnist_train = pd.read_csv(os.path.join(Data_Dir, Mnist_train_data_filename), header=None)
    df_mnist_test = pd.read_csv(os.path.join(Data_Dir, Mnist_test_data_filename), header=None)
    y_train = df_mnist_train.values[:, 0].astype('int')
    y_test = df_mnist_test.values[:, 0].astype('int')
    X_train_mnist = df_mnist_train.values[:, 1:].astype('int')
    if feature_select:
        X_train_mnist_resample = []
        for i in range(X_train_mnist.shape[0]):
            temp = np.reshape(X_train_mnist[i,:],[28,28])
            X_train_mnist_resample.append(hog(temp,block_norm='L2-Hys').flatten())
        X_train = np.asarray(X_train_mnist_resample)
    else:
        X_train=X_train_mnist

    X_test_mnist = df_mnist_test.values[:, 1:].astype('int')
    if feature_select:
        X_test_mnist_resample = []
        for i in range(X_test_mnist.shape[0]):
            temp = np.reshape(X_test_mnist[i, :], [28, 28])
            X_test_mnist_resample.append(hog(temp, block_norm='L2-Hys').flatten())
        X_test = np.asarray(X_test_mnist_resample)
    else:
        X_test=X_test_mnist

    return X_train,y_train,X_test,y_test

def Load_Abalone(Data_Dir,Data_filename):
    df = pd.read_csv(os.path.join(Data_Dir, Data_filename),
                     names=['S', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                            'Viscera weight',
                            'Shell weight', 'Rings'])
    data = df.values[:, 1:-1].astype('float')
    X_train = data[0:3000, :]
    X_test = data[3000:-1, :]
    target = df.values[:, -1]
    y_train = target[0:3000].astype('int32')
    y_test = target[3000:-1].astype('int32')
    return X_train, y_train, X_test, y_test
