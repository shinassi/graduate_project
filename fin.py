import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sp
import keras
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from pdb import set_trace
import csv

np.random.seed(0)

def generate_distance():
    # データ取得
    distance = np.loadtxt("/Users/al14046/mygrad/kari.csv",delimiter=",")
    input1 = distance[:,0:5]

    #正規化
    dataset = input1
    dataset[:,0] = dataset[:,0]/int(len(dataset))
    dataset[:,1] = dataset[:,1]/1980
    dataset[:,2] = dataset[:,2]/1080
    return dataset


def generate_data(distance, length_per_unit, dimension):
    sequences = []
    target = []
    num_classes = 10
    #for i in range(0, distance.size - length_per_unit):
    #    sequences.append(distance[i:i + length_per_unit])
    #    target.append(distance[i + length_per_unit])
    X = distance[:,0:4]
    Y = distance[:,4]
    #X = np.array(sequences).reshape(len(sequences), length_per_unit, dimension)
    #Y = np.array(target).reshape(len(sequences), dimension)

    N_train = int(len(distance) * 0.66)
    X_train = X[:N_train]
    X_validation = X[N_train:]
    Y_train = Y[:N_train]
    Y_validation = Y[N_train:]
    return (X_train, X_validation, Y_train, Y_validation)


def build_model(input_shape, hidden_layer_count):
    model = Sequential()
    model.add(Embedding(input_dim=DIMENSION, output_dim=100, init='glorot_uniform'))
    model.add(SimpleRNN(20, return_sequences=False))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])

    return model


# 一つの時系列データの長さ
LENGTH_PER_UNIT = 1 #変更可
# 一次元データを扱う
DIMENSION = 5
# データの生成
distance = generate_distance()
# トレーニング、バリデーション用データの生成
X_train, X_validation, Y_train, Y_validation = generate_data(distance, LENGTH_PER_UNIT, DIMENSION)

# SimpleRNN隠れ層の数
HIDDEN_LAYER_COUNT = 30
# 入力の形状
input_shape=(LENGTH_PER_UNIT, DIMENSION)
# モデルの生成
model = build_model(input_shape, HIDDEN_LAYER_COUNT)

# モデルのトレーニング
epochs = 100
batch_size = 20
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_validation, Y_validation),
          callbacks=[early_stopping])

 # モデルを評価
loss = model.evaluate(X_validation, Y_validation, verbose=0)
print('Test loss: %s '%(loss))
# 訓練データに対してモデルで出力を予測
# 0.0-1.0の確率値で出力されるため0.5以上の場合はクラス1、未満の場合は0と判定する
predictions = np.round((model.predict(X_validation)*10)) #予測値
correct = Y_validation[:, np.newaxis] #正解の値

print(predictions)
print(correct)
np.savetxt("bar.csv", model.predict(X_validation), delimiter=",")
