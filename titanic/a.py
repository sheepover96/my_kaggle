#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 16:10:43 2018

@author: miyamototatsurou
"""
import pickle
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
from keras.utils import np_utils
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical

from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import xgboost as xgb

from sklearn.svm import SVC #svm
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayse
from sklearn import tree #decision tree
from sklearn.ensemble import RandomForestClassifier #random forest
from sklearn.ensemble import AdaBoostClassifier #AdaBoost
from sklearn.linear_model import LogisticRegression #ロジスティック回帰
from sklearn.neural_network import MLPClassifier#Newral Network

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn import datasets
import numpy as np
import pandas as pd



def kesson_table(df): 
        null_val = df.isnull().sum()
        percent = 100 * df.isnull().sum()/len(df)
        kesson_table = pd.concat([null_val, percent], axis=1)
        kesson_table_ren_columns = kesson_table.rename(
        columns = {0 : '欠損数', 1 : '%'})
        return kesson_table_ren_columns


def pre_process(train, test):
    #dataの欠損値を補完
    train["Age"] = train["Age"].fillna(train["Age"].median())
    train["Embarked"] = train["Embarked"].fillna("S")
    test["Age"] = test["Age"].fillna(test["Age"].median())
    test['Fare'] = test['Fare'].fillna(test['Fare'].median())

    #dataの文字列要素を数値で置き換える
    train["Sex"][train["Sex"] == "male"] = 0
    train["Sex"][train["Sex"] == "female"] = 1
    train["Embarked"][train["Embarked"] == "S" ] = 0
    train["Embarked"][train["Embarked"] == "C" ] = 1
    train["Embarked"][train["Embarked"] == "Q"] = 2
    test["Sex"][test["Sex"] == "male"] = 0
    test["Sex"][test["Sex"] == "female"] = 1
    test["Embarked"][test["Embarked"] == "S"] = 0
    test["Embarked"][test["Embarked"] == "C"] = 1
    test["Embarked"][test["Embarked"] == "Q"] = 2

    return train, test


def SVM(x_train, y_train):
    print('model : SVM\n\n')
    #clf = GridSearchCV(SVC(), parameters)
    #clf.fit(x_train, y_train)
    
    # train SVC with searched paramters
    #model = clf.best_estimator_
    
    model = SVC(random_state=None, kernel='rbf')
    model.fit(x_train, y_train)
    
    
    return model, 'SVC.model'
    
    
def NB(x_train, y_train):
    print('model : Naive bayse\n\n')
    model = GaussianNB() # 正規分布を仮定したベイズ分類
    model.fit(x_train, y_train) # 学習をする

    return model, 'NB.model'  

def DT(x_train, y_train):
    print('model : Decision Tree\n\n')
    model = tree.DecisionTreeClassifier(max_depth=3)
    model = model.fit(x_train, y_train)

    return model, 'DT.model'

def RF(x_train, y_train):
    print('model : Random Forest\n\n')
    #model = RandomForestClassifier(min_samples_leaf=3, random_state=0)
    model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=25, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=15,
            min_weight_fraction_leaf=0.0, n_estimators=51, n_jobs=4,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
    model.fit(x_train, y_train)

    return model, 'RF.model'

def AdaBoost(x_train, y_train):
    print('model : AdaBoost\n\n')
    model = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
    model.fit(x_train, y_train)

    return model, 'AdaBoost.model'

def LR(x_train, y_train):
    print('model : Logistic Regression')
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    return model, 'LR.model'



def NN(x_train, y_train):
    print('model : Newral Network')
    model = MLPClassifier(solver="adam",random_state=None, learning_rate_init=0.1,
                          max_iter=200)
    #MLPClassifier(hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’,
                  #alpha=0.0001, batch_size=’auto’, learning_rate=’constant’, 
                  #learning_rate_init=0.001,random_state=None, tol=0.0001, verbose=False, 
                  #warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                  #early_stopping=False, validation_fraction=0.1, beta_1=0.9, 
                  #beta_2=0.999, epsilon=1e-08)[source]
    model.fit(x_train, y_train)

    return model, 'NN.model'


def DNN(x_train, y_train):
    print('model : Newral Network')
    model = Sequential()
    model.add(Dense(64, input_shape=(7,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(0.5))
    model.add(Activation('sigmoid'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    y_train = to_categorical(y_train)
    optimizer = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=8)

    return model, 'DNN.model'


def XGB(x_train, y_train):
    print('model : XGBoost')
    model = xgb.XGBClassifier()
    #MLPClassifier(hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’,
                  #alpha=0.0001, batch_size=’auto’, learning_rate=’constant’, 
                  #learning_rate_init=0.001,random_state=None, tol=0.0001, verbose=False, 
                  #warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                  #early_stopping=False, validation_fraction=0.1, beta_1=0.9, 
                  #beta_2=0.999, epsilon=1e-08)[source]
    model.fit(x_train, y_train)

    return model, 'NN.model'


if __name__ == '__main__':

    train = pd.read_csv("all/train.csv")
    test = pd.read_csv("all/test.csv")

    #前処理
    train, test = pre_process(train, test)

    # 「train」の目的変数と説明変数の値を取得
    target = train["Survived"].values
    train['FamilySize'] = train['SibSp'] + train['Parch']
    train['Title'] = 4
    train.loc[train["Name"].str.contains("Mr"), "Title"] = 0
    train.loc[train["Name"].str.contains("Miss"), "Title"] = 1
    train.loc[train["Name"].str.contains("Mrs"), "Title"] = 2
    train.loc[train["Name"].str.contains("Master"), "Title"] = 3
    #features_one = train[["Pclass", "Sex", "Age", "Fare"]].values
    features_one = train[["Pclass","Age","Sex","Fare", 'FamilySize', "Embarked", 'Title']].values

    with open('all_train_df.pickle', mode='rb') as f1:
        train2 = pickle.load(f1)
    train_label = train2['Label'].values
    train_data = train[['SVM', 'NB', 'DT', 'RF', 'AdaBoost', 'LR', 'NN']]

     # 乱数を制御するパラメータ random_state は None にすると毎回異なるデータを生成する
    x_train, x_test, y_train, y_test = train_test_split(train_data, target, test_size=0.3, random_state=None )

    # データの標準化処理
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)
    
    #x_train_std = x_train
    #x_test_std = x_test


    print('===================training classifier==================')
    model1, model_name1 = SVM(x_train_std, y_train) #svmで分類
    model2, model_name = NB(x_train_std, y_train) #Gaussian Naive Bayseで分類
    model3, model_name = DT(x_train_std, y_train) #Decision Tree
    model4, model_name = RF(x_train_std, y_train) #Random Forest
    model5, model_name = AdaBoost(x_train_std, y_train) #AdaBoost
    model6, model_name = LR(x_train_std, y_train) #Logistic Regression
    model7, model_name = NN(x_train_std, y_train) #Newral Network
    model8, model_name2 = XGB(x_train_std, y_train)
    model9, model_name = DNN(x_train_std, y_train) #Newral Network


    
    
    print('====================result=====================')
    # トレーニングデータに対する精度
    #pred_train = model.predict(x_train_std)
    #print(pred_train)
    #accuracy_train = accuracy_score(y_train, pred_train)
    #print('トレーニングデータに対する正解率： %.2f' % accuracy_train)
    
    # テストデータに対する精度
    pred_test = [i for i in range(100)]
    pred_test[1] = model1.predict(x_test_std)
    pred_test[2] = model2.predict(x_test_std)
    pred_test[3] = model3.predict(x_test_std)
    pred_test[4] = model4.predict(x_test_std)
    pred_test[5] = model5.predict(x_test_std)
    pred_test[6] = model6.predict(x_test_std)
    pred_test[7] = model7.predict(x_test_std)
    pred_test[8] = model8.predict(x_test_std)
    pred_test[9] = model9.predict_classes(x_test_std)

    accuracy_test_or = accuracy_score(y_test, pred_test[9])
    print('テストデータに対する正解率： %.2f' % accuracy_test_or)
    accuracy_test_and = accuracy_score(y_test, np.logical_and(pred_test[1], pred_test[9]))
    print('テストデータに対する正解率： %.2f' % accuracy_test_and)


    # 「test」の説明変数の値を取得
    #test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

    test['FamilySize'] = test['SibSp'] + test['Parch']
    test['Title'] = 4
    test.loc[test["Name"].str.contains("Mr"), "Title"] = 0
    test.loc[test["Name"].str.contains("Miss"), "Title"] = 1
    test.loc[test["Name"].str.contains("Mrs"), "Title"] = 2
    test.loc[test["Name"].str.contains("Master"), "Title"] = 3
    test_features = test[["Pclass","Age","Sex","Fare", 'FamilySize', "Embarked", 'Title']].values
    
    # データの標準化処理
    sc = StandardScaler()
    sc.fit(test_features)
    test_features_std  = sc.transform(test_features)
    
    
    # 「test」の説明変数を使ってモデルで予測
    pred = model9.predict_classes(test_features_std)
    #pred = np.where(pred==True,1,pred)
    #pred = np.where(pred==False,0,pred)

    
    #予測データの中身を確認
    print(pred)
    print(len(pred))
    
    
    
    # PassengerIdを取得
    PassengerId = np.array(test["PassengerId"]).astype(int)
    # my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
    my_solution = pd.DataFrame(pred, PassengerId, columns = ["Survived"])
    # my_tree_one.csvとして書き出し
    my_solution.to_csv("my_tree_one_xgb.csv", index_label = ["PassengerId"])
    
    
    #
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
