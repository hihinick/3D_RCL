import os
import re
import gc
import cv2
import time
import torch
import random
import one_D_RBF
import two_D_RBF_v2 as two_D_RBF
import plotly.graph_objs as go
import plotly.offline as py
py.offline.init_notebook_mode()
import plot_v2 as plot
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import trange
from random import randrange
from datetime import date, datetime, timedelta

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, Sequential, activations
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Flatten,Conv2D,BatchNormalization,Activation,GlobalAveragePooling2D,concatenate, AveragePooling2D, MaxPool2D, MaxPooling2D, Dense, Input, Concatenate,TimeDistributed,Lambda,SimpleRNN,LSTM,Dropout,Permute,Reshape 
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import L1L2

class three_D_RCL():
    def __init__(self, simulate_area, group, model_type, look_back, pre_time, model_NO, **kwargs):
        self.simulate_area = simulate_area
        self.group = group
        self.model_type = model_type
        self.look_back = look_back
        self.pre_time = pre_time
        self.model_NO = model_NO
        
    def test(self):
        #讀檔案
        base_dir = 'C:/Users/User/0000.碩論資料/3D_RCL'
        target_dir = base_dir + '/{}'.format('data')
        one_D_data = np.load(target_dir+'/one_D_data.npy')
        two_D_data = np.load(target_dir+'/area{}_group{}.npy'.format(self.simulate_area, self.group))
        target = np.load(target_dir+'/area{}_group{}_sum.npy'.format(self.simulate_area, self.group))
#         print('=============================================')
#         print('讀進來的資料shape')
#         print('one_D_data', np.shape(one_D_data))
#         print('two_D_data', np.shape(two_D_data))
#         print('target', np.shape(target))
        # 正規化
        one_D_data_norm = self.each_norm(one_D_data)
        two_D_data_norm = self.global_norm(two_D_data)
        target_max = target.max()
        target_min = target.min()
        target_norm = (target-target_min)/(target_max-target_min)
#         print('=============================================')
#         print('正規化後最大最小值')
#         print('one_D_data', np.shape(one_D_data_norm), one_D_data_norm.max(), one_D_data_norm.min())
#         print('two_D_data', np.shape(two_D_data_norm), two_D_data_norm.max(), two_D_data_norm.min())
#         print('target', np.shape(target_norm), target_norm.max(), target_norm.min())
        # 建立時間序列
        One_D_data_norm, Two_D_data_norm, Target_norm = self.bulid_time_series(self.look_back, self.pre_time, one_D_data_norm, two_D_data_norm, target_norm)
        shape = np.shape(Two_D_data_norm)
        Two_D_data_norm = np.reshape(Two_D_data_norm, (shape[0], shape[1], shape[2], shape[3], 1))
        # 切分訓練測試
        One_D_data_norm = One_D_data_norm.astype('float64')
        Two_D_data_norm = Two_D_data_norm.astype('float64')
        Target_norm = Target_norm.astype('float64')

        test_size = 0.15
        one_D_train, one_D_test = self.train_test_split(One_D_data_norm, test_size)
        two_D_train, two_D_test = self.train_test_split(Two_D_data_norm, test_size)
        target_train, target_test = self.train_test_split(Target_norm, test_size)
#         print('=============================================')
#         print('訓練測試集的shape')
#         print(np.shape(one_D_train), np.shape(one_D_test))
#         print(np.shape(two_D_train), np.shape(two_D_test))
#         print(np.shape(target_train), np.shape(target_test))
        stationQTY = np.shape(one_D_train)[1]
        # 檢查缺失值
#         print('=============================================')
#         print('是否有缺失值')
#         print(np.isnan(np.sum(one_D_train)))
#         print(np.isnan(np.sum(two_D_train)))
#         print(np.isnan(np.sum(target_norm)))
        # 進模型
        tf.data.experimental.enable_debug_mode()
#         print('=============================================')
#         print('進模型')
#         print(one_D_train.max(), one_D_test.min())
#         print(two_D_train.max(), two_D_test.min())
#         print(target_train.max(), target_test.min())
        stationQTY = np.shape(one_D_train)[2]
        
        look_back = np.shape(two_D_train)[1]
        Width = np.shape(two_D_train)[2]
        Length  = np.shape(two_D_train)[3]
        Channel = np.shape(two_D_train)[4]
        Model_type = self.model_type
#         print('=============================================')
#         print('model type=', Model_type)
        if Model_type == '1LSTM':
            choose_model = self.LSTM_1(Channel, look_back, Width, Length, stationQTY)
            tf.config.run_functions_eagerly(self.LSTM_1)
        if Model_type == '2LSTM':
            choose_model = self.LSTM_2(Channel, look_back, Width, Length, stationQTY)
            tf.config.run_functions_eagerly(self.LSTM_2)
        if Model_type == '1RNN':
            choose_model = self.RNN_1(Channel, look_back, Width, Length, stationQTY)
            tf.config.run_functions_eagerly(self.RNN_1)
#         Fig_name_emp, RMSE_emp, MSE_emp, MAE_emp, MAPE_emp  = [], [], [], [], []

        choose_model.load_weights(r"C:\Users\User\0000.碩論資料\3D_RCL\simulate_area{}_group{}\best_models_{}_look_back{}\best_model_{}.h5".format(self.simulate_area, self.group, self.model_type, self.look_back, self.model_NO))
        oneD_test = one_D_test.copy()
        twoD_test = two_D_test.copy()
        Target_test = target_test.copy()

        output = choose_model.predict({'2D_input':twoD_test, '1D_input':oneD_test})
        ans = output*(target_max-target_min)+target_min
        Target_test = Target_test*(target_max-target_min)+target_min

        RMSE = round(np.sqrt(mean_squared_error(ans, Target_test)), 3)
        MSE = round((np.sqrt(mean_squared_error(ans, Target_test)))**2, 3)
        MAE = round(float(mean_absolute_error(np.array(ans).flatten(), np.array(Target_test).flatten())), 3)
        MAPE = round(np.mean(np.abs((ans-Target_test)/Target_test))*100, 3)
        print('train size: ',np.shape(one_D_train)[0])
        print('test size: ',np.shape(one_D_test)[0])
        print('RMSE：', RMSE)
        print('MSE：', MSE)
        print('MAE:', MAE)
        print('MAPE:', MAPE)
        #視覺化預測跟實際的值
        fig_name = 'area{}_gropu{}, {}_model{}, RMSE={}'.format(self.simulate_area, self.group, self.model_type, self.model_NO, RMSE)
        pre_dataframe = pd.DataFrame(ans,columns=['pre'])
        compare = pd.concat([pd.DataFrame(Target_test, columns=['act']), pre_dataframe],axis=1)
        show = plot.mat_plot(compare, x_name='time', y_name='crowded', title=fig_name, Save=True, fig_name=fig_name)
        show.plotly_plot()
#         Fig_name_emp.append(fig_name)
#         RMSE_emp.append(RMSE)
#         MSE_emp.append(MSE)
#         MAE_emp.append(MAE)
#         MAPE_emp.append(MAPE)
#         return fig_name_emp, RMSE_emp, MSE_emp, MAE_emp, MAPE_emp
        return fig_name, RMSE, MSE, MAE, MAPE
        
    def train(self):
        #讀檔案
        base_dir = 'C:/Users/User/0000.碩論資料/3D_RCL'
        target_dir = base_dir + '/{}'.format('data')
        one_D_data = np.load(target_dir+'/one_D_data.npy')
        two_D_data = np.load(target_dir+'/area{}_group{}.npy'.format(self.simulate_area, self.group))
        target = np.load(target_dir+'/area{}_group{}_sum.npy'.format(self.simulate_area, self.group))
        print('=============================================')
        print('讀進來的資料shape')
        print('one_D_data', np.shape(one_D_data))
        print('two_D_data', np.shape(two_D_data))
        print('target', np.shape(target))
        # 正規化
        one_D_data_norm = self.each_norm(one_D_data)
        two_D_data_norm = self.global_norm(two_D_data)
        target_max = target.max()
        target_min = target.min()
        target_norm = (target-target_min)/(target_max-target_min)
        print('=============================================')
        print('正規化後最大最小值')
        print('one_D_data', np.shape(one_D_data_norm), one_D_data_norm.max(), one_D_data_norm.min())
        print('two_D_data', np.shape(two_D_data_norm), two_D_data_norm.max(), two_D_data_norm.min())
        print('target', np.shape(target_norm), target_norm.max(), target_norm.min())
        # 建立時間序列
        One_D_data_norm, Two_D_data_norm, Target_norm = self.bulid_time_series(self.look_back, self.pre_time, one_D_data_norm, two_D_data_norm, target_norm)
        shape = np.shape(Two_D_data_norm)
        Two_D_data_norm = np.reshape(Two_D_data_norm, (shape[0], shape[1], shape[2], shape[3], 1))
        # 切分訓練測試
        One_D_data_norm = One_D_data_norm.astype('float64')
        Two_D_data_norm = Two_D_data_norm.astype('float64')
        Target_norm = Target_norm.astype('float64')

        test_size = 0.15
        one_D_train, one_D_test = self.train_test_split(One_D_data_norm, test_size)
        two_D_train, two_D_test = self.train_test_split(Two_D_data_norm, test_size)
        target_train, target_test = self.train_test_split(Target_norm, test_size)
        print('=============================================')
        print('訓練測試集的shape')
        print(np.shape(one_D_train), np.shape(one_D_test))
        print(np.shape(two_D_train), np.shape(two_D_test))
        print(np.shape(target_train), np.shape(target_test))
        stationQTY = np.shape(one_D_train)[1]
        # 檢查缺失值
        print('=============================================')
        print('是否有缺失值')
        print(np.isnan(np.sum(one_D_train)))
        print(np.isnan(np.sum(two_D_train)))
        print(np.isnan(np.sum(target_norm)))
        # 進模型
        tf.data.experimental.enable_debug_mode()
        print('=============================================')
        print('進模型')
        print(one_D_train.max(), one_D_test.min())
        print(two_D_train.max(), two_D_test.min())
        print(target_train.max(), target_test.min())
        stationQTY = np.shape(one_D_train)[2]
        
        look_back = np.shape(two_D_train)[1]
        Width = np.shape(two_D_train)[2]
        Length  = np.shape(two_D_train)[3]
        Channel = np.shape(two_D_train)[4]
        Model_type = self.model_type
        print('=============================================')
        print('model type=', Model_type)
        if Model_type == '1LSTM':
            choose_model = self.LSTM_1(Channel, look_back, Width, Length, stationQTY)
            tf.config.run_functions_eagerly(self.LSTM_1)
        if Model_type == '2LSTM':
            choose_model = self.LSTM_2(Channel, look_back, Width, Length, stationQTY)
            tf.config.run_functions_eagerly(self.LSTM_2)
        if Model_type == '1RNN':
            choose_model = self.RNN_1(Channel, look_back, Width, Length, stationQTY)
            tf.config.run_functions_eagerly(self.RNN_1)
       
        def_run = self.run(base_dir, self.model_NO, choose_model, two_D_train, one_D_train, target_train)
        del def_run, choose_model
        gc.collect()
            
    # npy 一般正規化(each_column)
    def each_norm(self, data):
        original = data.copy()
        data = pd.DataFrame(original)
        cnames = list(data.columns)
        for cname in cnames :
            data[cname] = (data[cname] - data[cname].min()) / (data[cname].max() - data[cname].min())
        data = np.array(data)
        return data
    
    # npy 全域正規化(all_columns)
    def global_norm(self, stage):
        min_ = stage.min()
        max_ = stage.max()
        gap = (max_-min_)
        norm = (stage-min_)/gap
        norm = norm
        return norm
    
    def bulid_time_series(self, look_back, pre_time, one_D_data_norm, two_D_data_norm, target_norm):
        one_D = one_D_data_norm.copy()
        one_D_shape = np.shape(one_D)
        one_D_empty = np.ones((one_D_shape[0]-look_back-pre_time+1, look_back, one_D_shape[1]))

        two_D = two_D_data_norm.copy()
        two_D_shape = np.shape(two_D)
        two_D_empty = np.ones((two_D_shape[0]-look_back-pre_time+1, look_back, two_D_shape[1], two_D_shape[2]))

        tar = target_norm.copy()
        tar_shape = np.shape(tar)
        tar_empty = np.ones((tar_shape[0]-look_back-pre_time+1, pre_time))

        for i in range(tar_shape[0]-look_back-pre_time+1):
            one_D_empty[i] = one_D[i: look_back+i]
            two_D_empty[i] = two_D[i: look_back+i]
            tar_empty[i] = tar[look_back+i: pre_time+look_back+i]

        return one_D_empty, two_D_empty, tar_empty

    def train_test_split(self, data, test_size):
        lenth = len(data)
        split = int(lenth*(1-test_size))
        train = data[:split]
        test = data[split:]
        return train, test      

    def LSTM_1(self, Channel, look_back, Width, Length, stationQTY):
        Two_D_input = Input(shape=(look_back, Width, Length, Channel), name='2D_input')
        Map_in = Permute((1, 4, 2, 3))(Two_D_input)
        Map = Reshape((look_back*Channel, Width, Length))(Map_in)
        Map = two_D_RBF.FuzzyLayer(look_back*Channel*3, count_W=Width, count_L=Length, window_count=look_back*Channel, name='2D_RBF')(Map)
        Map = Reshape((look_back, Channel*Length*Width, Length*Width))(Map)
        Map = Permute((1, 3, 2))(Map)
        Map = Reshape((look_back, Width, Length, Channel*Length*Width))(Map)

        Map = TimeDistributed(Conv2D(16, (3,3), activation='relu'))(Map)
        Map = TimeDistributed(Conv2D(8, (3,3), activation='relu'))(Map)
        Map = TimeDistributed(BatchNormalization())(Map)
        Map = TimeDistributed(GlobalAveragePooling2D())(Map)
        Map = BatchNormalization()(Map)
        Map = Reshape((look_back, 8))(Map) #permute後，變成12(look_back) * 81(9*9攤平) => 後面的是要放被篩選的

        One_D_input = Input(shape=(look_back, stationQTY, 1), name='1D_input')
        Multi_in = Reshape((look_back, stationQTY))(One_D_input)
        Merge = concatenate([Multi_in, Map], axis=2)

        Merge = Reshape((look_back*(stationQTY+8), ))(Merge)

        Main = one_D_RBF.FuzzyLayer(fuzzy_size=3, input_dim=stationQTY+8, name='1D_RBF')(Merge)

        Main = Dense(32)(Main)
        Main = LSTM(16)(Main)
        Main = Dense(1, activation='relu', name='dense_out')(Main)
        model = Model(inputs=[Two_D_input,One_D_input], outputs=Main)
        return model        

    def LSTM_2(self, Channel, look_back, Width, Length, stationQTY):
        Two_D_input = Input(shape=(look_back, Width, Length, Channel), name='2D_input')
        Map_in = Permute((1, 4, 2, 3))(Two_D_input)
        Map = Reshape((look_back*Channel, Width, Length))(Map_in)
        Map = two_D_RBF.FuzzyLayer(look_back*Channel*3, count_W=Width, count_L=Length, window_count=look_back*Channel, name='2D_RBF')(Map)
        Map = Reshape((look_back, Channel*Length*Width, Length*Width))(Map)
        Map = Permute((1, 3, 2))(Map)
        Map = Reshape((look_back, Width, Length, Channel*Length*Width))(Map)

        Map = TimeDistributed(Conv2D(16, (3,3), activation='relu'))(Map)
        Map = TimeDistributed(Conv2D(8, (3,3), activation='relu'))(Map)
        Map = TimeDistributed(BatchNormalization())(Map)
        Map = TimeDistributed(GlobalAveragePooling2D())(Map)
        Map = BatchNormalization()(Map)
        Map = Reshape((look_back, 8))(Map) #permute後，變成12(look_back) * 81(9*9攤平) => 後面的是要放被篩選的

        One_D_input = Input(shape=(look_back, stationQTY, 1), name='1D_input')
        Multi_in = Reshape((look_back, stationQTY))(One_D_input)
        Merge = concatenate([Multi_in, Map], axis=2)

        Merge = Reshape((look_back*(stationQTY+8), ))(Merge)

        Main = one_D_RBF.FuzzyLayer(fuzzy_size=3, input_dim=stationQTY+8, name='1D_RBF')(Merge)

        Main = Dense(32)(Main)
        Main = LSTM(32, return_sequences=True)(Main)
        Main = LSTM(16)(Main)
        Main = Dense(1, activation='relu', name='dense_out')(Main)
        model = Model(inputs=[Two_D_input,One_D_input], outputs=Main)
        return model        

    def RNN_2(self, Channel, look_back, Width, Length, stationQTY):
        Two_D_input = Input(shape=(look_back, Width, Length, Channel), name='2D_input')
        Map_in = Permute((1, 4, 2, 3))(Two_D_input)
        Map = Reshape((look_back*Channel, Width, Length))(Map_in)
        Map = two_D_RBF.FuzzyLayer(look_back*Channel*3, count_W=Width, count_L=Length, window_count=look_back*Channel, name='2D_RBF')(Map)
        Map = Reshape((look_back, Channel*Length*Width, Length*Width))(Map)
        Map = Permute((1, 3, 2))(Map)
        Map = Reshape((look_back, Width, Length, Channel*Length*Width))(Map)

        Map = TimeDistributed(Conv2D(16, (3,3), activation='relu'))(Map)
        Map = TimeDistributed(Conv2D(8, (3,3), activation='relu'))(Map)
        Map = TimeDistributed(BatchNormalization())(Map)
        Map = TimeDistributed(GlobalAveragePooling2D())(Map)
        Map = BatchNormalization()(Map)
        Map = Reshape((look_back, 8))(Map) #permute後，變成12(look_back) * 81(9*9攤平) => 後面的是要放被篩選的

        One_D_input = Input(shape=(look_back, stationQTY, 1), name='1D_input')
        Multi_in = Reshape((look_back, stationQTY))(One_D_input)
        Merge = concatenate([Multi_in, Map], axis=2)

        Merge = Reshape((look_back*(stationQTY+8), ))(Merge)

        Main = one_D_RBF.FuzzyLayer(fuzzy_size=3, input_dim=stationQTY+8, name='1D_RBF')(Merge)

        Main = Dense(32)(Main)
        Main = SimpleRNN(32, activation=None, return_sequences=True)(Main)
        Main = SimpleRNN(32, activation=None)(Main)
        Main = Dense(1, activation='relu', name='dense_out')(Main)
        model = Model(inputs=[Two_D_input,One_D_input], outputs=Main)
        return model        

    def run(self, base_dir, model_NO, choose_model, two_D_train, one_D_train, target_train):
        model = choose_model
        target_dir = base_dir + '/simulate_area{}_group{}'.format(self.simulate_area, self.group)
        tf.data.experimental.enable_debug_mode()
        tf.keras.backend.clear_session()
        if not os.path.isdir(os.path.abspath(target_dir + '/best_models_{}_look_back{}'.format(self.model_type, self.look_back))):
            os.makedirs(os.path.abspath(target_dir + '/best_models_{}_look_back{}'.format(self.model_type, self.look_back)))

        ###用時間來當記錄模型的檔名
        checkpoint = ModelCheckpoint(target_dir+'/best_models_{}_look_back{}/best_model_{}.h5'.format(self.model_type, self.look_back, self.model_NO), monitor='val_loss',  save_best_only=True)

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0)
        if True:
            Freeze_epoch = 0
            Epoch = 100
            batch_size = 20
            learning_rate_base = 1e-4
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate_base), loss='mse', metrics=['mae']) # 迴歸
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(two_D_train), len(two_D_train)*0.2, batch_size))

            model.fit({'2D_input':two_D_train, '1D_input':one_D_train}, {'dense_out':target_train},
                      batch_size = batch_size, 
                      validation_split = 0.2, 
                      epochs = Epoch,
                      initial_epoch = Freeze_epoch,
                      callbacks = [reduce_lr, early_stopping, checkpoint],
                      verbose=1, shuffle=False)
        #訓練曲線圖 回歸
        train_loss = model.history.history['loss']
        val_loss = model.history.history['val_loss']
        epochs = range(1, len(train_loss)+1)
        plt.plot(epochs, train_loss, 'b-', label='train_loss', color='blue')
        plt.plot(epochs, val_loss, 'b-', label='val_loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
        del model