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
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from typing import List
import requests
import os

from flask import Flask
from flask import request, jsonify, render_template, redirect
import sys

from abc import ABC, abstractmethod

app=Flask(__name__, template_folder='../templates')

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

class Strategy(ABC):
    @abstractmethod
    def trainml(self):
        pass
    
    @abstractmethod
    def anomalydetection(self):
        pass
    
class Context():
    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def logic(self) -> None:
        print("===========================================")
        print("TrainML is done")
        self._strategy.trainml()
        print("Anomaly detection is done")
        self._strategy.anomalydetection()
        print("===========================================")    
        
class randomforest(Strategy):
    def __init__(self) -> None:
        self.kfold = None
        self.paths = '../dataset/process_data.csv'
        
        df = pd.read_csv(self.paths)
        y = df['fraud']
        X =  df.loc[:, df.columns != 'fraud']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    def anomalydetection(self):
        print("Concrete Component(RandomForest) - AnomalyDetection")
        modelcv = RandomForestClassifier(random_state=42)
        print("===========================================")
        print("Classifier is ready")
        scoring = 'accuracy'
        results = model_selection.cross_val_score(modelcv, self.x_train, self.y_train, cv = self.kfold, scoring = scoring)
        print("===========================================")
        print("10-fold cross validation average accuracy of the random forest model: %.3f" % (results.mean()))
        return jsonify({'sucess': 200, "message":"randomforest is task is done", "cross validation average accuracy": results.mean()})
        
    def trainml(self):
        print("Concrete Component(RandomForest) -  trainML")
        try:
            rf = RandomForestClassifier(random_state=42)
            rf.fit(self.x_train, self.y_train)
            print("===========================================")
            print('Random Forest accuracy: {:.3f}'.format(accuracy_score(self.y_test, rf.predict(self.x_test))))
            self.kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle = True)
            print("===========================================")
            print("Kfold is ready")
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})

class supportvectormachine(Strategy):
    def __init__(self) -> None:
        self.kfold = None
        df = pd.read_csv('../dataset/process_data.csv')
        y = df['fraud']
        X =  df.loc[:, df.columns != 'fraud']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    def anomalydetection(self):

        print("Concrete Component(SupportVectorMachine) - AnomalyDetection")
        modelCV = SVC(random_state=42)
        print("===========================================")
        print("SVC is ready")
        scoring = 'accuracy'
        results = model_selection.cross_val_score(modelCV, self.x_train, self.y_train, cv = self.kfold, scoring = scoring)
        print("===========================================")
        print("10-fold cross validation average accuracy of the support vector machine model: %.3f" % (results.mean()))
        return jsonify({'sucess': 200, "message":"svm is task is done", "cross validation average accuracy": results.mean()})
        
    def trainml(self):
        print("Concrete Component(SupportVectorMachine) - trainML")
        try:
            svc = SVC(random_state=42)
            svc.fit(self.x_train, self.y_train)
            print("===========================================")
            print('Support vector machine accuracy: {:.3f}'.format(accuracy_score(self.y_test, svc.predict(self.x_test))))

            self.kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle = True)
            print("===========================================")
            print("Kfold is ready")
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})

class logisticregression(Strategy):
    def __init__(self) -> None:
        self.kfold = None
        df = pd.read_csv('../dataset/process_data.csv')
        y = df['fraud']
        X =  df.loc[:, df.columns != 'fraud']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    def anomalydetection(self):
        print("Concrete Component(LogisticRegression) - AnomalyDetection")
        modellrg = LogisticRegression(random_state=42)
        print("===========================================")
        print("LogisticRegression is ready")
        scoring = 'accuracy'
        results = model_selection.cross_val_score(modellrg, self.x_train, self.y_train, cv = self.kfold, scoring = scoring)
        print("===========================================")
        print("10-fold cross validation average accuracy of the Logistic Regression model: %.3f" % (results.mean()))
        return jsonify({'sucess': 200, "message":"Logistic Regression is task is done", "cross validation average accuracy": results.mean()})

    def trainml(self):
        try:
            print("Concrete Component(LogisticRegression) - trainML")
            lrg = LogisticRegression(random_state=42)
            lrg.fit(self.x_train, self.y_train)
            print("===========================================")
            print('LogisticRegression machine accuracy: {:.3f}'.format(accuracy_score(self.y_test, lrg.predict(self.x_test))))
            kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle = True)
            print("===========================================")
            print("Kfold is ready")
            
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})


@app.route('/api/anomalydetection', methods=['GET'], endpoint = 'randomforest')
def randomforest_api_stragtegy():
    try:
        RF = request.args.get('rf')
        LG = request.args.get('lg')
        SM = request.args.get('sm')

        print("Aglorithms to run: ",RF,LG, SM)
        if RF == "RF":
            context = Context(randomforest())
            print("===========================================")
            print("Client: Strategy is set to randomforest.")
            context.logic()
        if SM == "SVM":
            print("===========================================")
            print("Client: Strategy is set to SupportVectorMachine.")
            context =  Context(supportvectormachine())
            context.logic()
        if LG == "LG":
            print("===========================================")
            print("Client: Strategy is set to Logistic Regression.")
            context = Context(logisticregression())
            context.logic()
        return redirect("http://localhost:5010/", code=302)
    except:
        e = sys.exc_info()[0]
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    PORT = os.environ.get('PORT', 5010)
    app.run(debug=True, host='0.0.0.0', port=PORT)
