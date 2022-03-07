from cgitb import reset
from dataclasses import dataclass
from urllib import response
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.linear_model import Ridge


from typing import List
import requests
import os

from flask import Flask
from flask import request, jsonify, render_template, redirect
import sys

from abc import ABC, abstractmethod

app=Flask(__name__, template_folder='../templates')


from abc import ABCMeta, abstractmethod


class Preprocessing(metaclass=ABCMeta):
    """
    Component
    """
    @abstractmethod
    def preprocess_file(self) -> int:
        pass

    @abstractmethod
    def selectfeaturing(self) -> str:
        pass

class ConcreteComponent(Preprocessing):
    """
    Concrete Component
    """
    def preprocess_file(self) -> int:
        return pd.read_csv('../dataset/frauddetection.csv')

    def selectfeaturing(self) -> str:
        return pd.read_csv('../dataset/frauddetection.csv')

class ProcessDecorator(Preprocessing):
    """
    Decorator
    """
    def __init__(self, preprocess: Preprocessing):
        self.__preprocess = preprocess

    @abstractmethod
    def preprocess_file(self) -> int:
        return self.__preprocess.preprocess_file()

    @abstractmethod
    def selectfeaturing(self) -> str:
        return self.__preprocess.selectfeaturing()

class LASSO(ProcessDecorator):
    """
    Concrete Decorator
    """
    def __init__(self, preprocess: Preprocessing):
        super().__init__(preprocess)
        self.data_x1 = None

    def selectfeaturing(self) -> int:
        cols = []
        df =  super().selectfeaturing()
        for i in range (0, len(self.data_x1['Importance'])):
            if self.data_x1['Importance'][i] == 1:
                cols.append(self.data_x1['Feature'][i])
        print(cols)
        result = pd.concat([df[cols], df['fraud']], axis=1)
        result.to_csv('../dataset/process_data.csv', encoding='utf-8', index=False)


    def preprocess_file(self) -> str:
        df = super().preprocess_file()
        self.data_x1 = df.columns.values.tolist()
        y = ['fraud']
        X = [i for i in self.data_x1 if i not in y]
        model = linear_model.Lasso(alpha=0.1)
        rfe = RFE(model)
        rfe = rfe.fit(df[X], df[y].values.ravel())
        self.data_x1 = pd.DataFrame({
        'Feature': df[X].columns,'Importance': rfe.ranking_},)

class RidegeRegression(ProcessDecorator):
    """
    Concrete Decorator
    """
    def __init__(self, preprocess: Preprocessing):
        super().__init__(preprocess)
        self.data_x1 = None

    def selectfeaturing(self) -> str:
        df = super().selectfeaturing()
        cols = []
        for i in range (0, len(self.data_x1['Importance'])):
            if self.data_x1['Importance'][i] == 1:
                cols.append(self.data_x1['Feature'][i])

        result = pd.concat([df[cols], df['fraud']], axis=1)
        result.to_csv('../dataset/process_data.csv', encoding='utf-8', index=False)

    
    def preprocess_file(self) -> str:
        df = super().preprocess_file()
        df_vars = df.columns.values.tolist()
        y = ['fraud']
        X = [i for i in df_vars if i not in y]
        model = Ridge(alpha=1.0)

        rfe = RFE(model)
        rfe = rfe.fit(df[X], df[y].values.ravel())
        self.data_x1 = pd.DataFrame({
        'Feature': df[X].columns,'Importance': rfe.ranking_},)


class PreProcessing:
    @app.route('/')
    @app.route('/index')
    def index():
        return render_template("index.html")

    @app.route('/api/preprocess/', methods=['GET'], endpoint = 'preprocess')
    def preprocess():
        try:
            flag = request.args.get('process')
            concrete = ConcreteComponent()
            if flag == "Lasso":
                print("===========================================")
                print("Client: Strategy is set to rfe_lasso.")

                preprocess_lasso = LASSO(concrete)
                preprocess_lasso.preprocess_file()
                preprocess_lasso.selectfeaturing()

            if flag == "RG":
                print("===========================================")
                print("Client: Strategy is set to selectfeaturing.")
                preprocess_lasso = RidegeRegression(concrete)
                preprocess_lasso.preprocess_file()
                preprocess_lasso.selectfeaturing()
            return redirect("http://localhost:5011/", code=302)
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    PORT = os.environ.get('PORT', 5011)
    app.run(debug=True, host='0.0.0.0', port=PORT)