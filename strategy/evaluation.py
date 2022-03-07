
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.svm import SVC

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
import matplotlib
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from flask import redirect,send_from_directory

import os

from flask import Flask
from flask import request, jsonify, render_template, redirect
import sys
from abc import ABC, abstractmethod

app=Flask(__name__, template_folder='../templates')


class Strategy(ABC):
    
    @abstractmethod
    def process(self):
        print("main, method called")
        pass

    @abstractmethod
    def evaluation(self):
        print("main, method called")
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
        print("===== Context: Calling evaluation on data using the strategy =====")
        result = self._strategy.process()
        result = self._strategy.evaluation()
        # print("Task:  ",",".join(result))

class ConcreteEvaluationHeatMap(Strategy):

    def __init__(self) -> None:
        self.logreg_cm = None
        self.forest_cm = None
        self.svc_cm = None
        self.logreg = None
        self.rf = None 
        self.svc = None
        self.x_test = None
        self.y_test = None


    def process(self):
        print ("===============  HeatMap process is started  =================")
        df = pd.read_csv('../dataset/process_data.csv')
        y = df['fraud']
        X =  df.loc[:, df.columns != 'fraud']
        x_train, self.x_test, y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Logistic Regression
        print ("xxxxxxxxxxxxxxx Load Logistic Regression xxxxxxxxxxxxxxx")
        self.logreg = LogisticRegression(random_state=42)
        self.logreg.fit(x_train, y_train)
        logreg_y_pred = self.logreg.predict(self.x_test)
        self.logreg_cm = confusion_matrix(logreg_y_pred, self.y_test, labels=[1,0])

        # Random Forest
        print ("xxxxxxxxxxxxxxx Load Random Forest xxxxxxxxxxxxxxx")
        self.rf = RandomForestClassifier(random_state=42)
        self.rf.fit(x_train, y_train)
        y_pred = self.rf.predict(self.x_test)
        self.forest_cm = confusion_matrix(y_pred, self.y_test, labels=[1,0])

        # Support Vector Machine
        print ("xxxxxxxxxxxxxxx Load Support Vector Machine xxxxxxxxxxxxxxx")
        self.svc = SVC(random_state=42)
        self.svc.fit(x_train, y_train)
        svc_y_pred = self.svc.predict(self.x_test)
        self.svc_cm = confusion_matrix(svc_y_pred, self.y_test, labels=[1,0])

    def evaluation(self):
        print("=====   Concrete Evaluation HeatMap   =======")
        try:        
            plt.clf()
            sns.heatmap(self.logreg_cm, cmap="RdPu", annot=True, fmt=".0f",xticklabels = ["Fraudulent", "Legitimate"], yticklabels = ["Fraudulent", "Legitimate"])
            plt.ylabel("True class")
            plt.xlabel("Predicted class")
            plt.title("Logistic Regression")
            plt.savefig("logistic_regression")

            plt.clf()
            sns.heatmap(self.forest_cm, cmap="RdPu", annot=True, fmt=".0f",xticklabels = ["Fraudulent", "Legitimate"], yticklabels = ["Fraudulent", "Legitimate"])
            plt.ylabel("True class")
            plt.xlabel("Predicted class")
            plt.title("Random Forest")
            plt.savefig("random_forest")
            
            plt.clf()
            sns.heatmap(self.svc_cm, cmap="RdPu", annot=True, fmt=".0f",xticklabels = ["Fraudulent", "Legitimate"], yticklabels = ["Fraudulent", "Legitimate"])
            plt.ylabel("True class")
            plt.xlabel("Predicted class")
            plt.title("Support Vector Machine")
            plt.savefig("support_vector_machine")

            # return send_from_directory("./", filename="logistic_regression.png", as_attachment=True)
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})


class ConcreteEvaluationResult(Strategy):
    
    def __init__(self):
        self.logreg_cm = None
        self.forest_cm = None
        self.svc_cm = None
        self.logreg = None
        self.rf = None 
        self.svc = None
        self.x_test = None
        self.y_test = None

    def process(self):
        print ("===============  Print process is started  =================")
        df = pd.read_csv('../dataset/process_data.csv')
        y = df['fraud']
        X =  df.loc[:, df.columns != 'fraud']
        x_train, self.x_test, y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Logistic Regression
        print ("xxxxxxxxxxxxxxx Load Logistic Regression xxxxxxxxxxxxxxx")
        self.logreg = LogisticRegression(random_state=42)
        self.logreg.fit(x_train, y_train)
        logreg_y_pred = self.logreg.predict(self.x_test)
        self.logreg_cm = confusion_matrix(logreg_y_pred, self.y_test, labels=[1,0])

        # Random Forest
        print ("xxxxxxxxxxxxxxx Load Random Forest xxxxxxxxxxxxxxx")
        self.rf = RandomForestClassifier(random_state=42)
        self.rf.fit(x_train, y_train)
        y_pred = self.rf.predict(self.x_test)
        self.forest_cm = confusion_matrix(y_pred, self.y_test, labels=[1,0])

        # Support Vector Machine
        print ("xxxxxxxxxxxxxxx Load Support Vector Machine xxxxxxxxxxxxxxx")
        self.svc = SVC(random_state=42)
        self.svc.fit(x_train, y_train)
        svc_y_pred = self.svc.predict(self.x_test)
        self.svc_cm = confusion_matrix(svc_y_pred, self.y_test, labels=[1,0])

    def evaluation(self):
        print("=====   Concrete Evaluation Result   =======")
        print("\033[1m The result is telling us that we have: ",(self.logreg_cm[0,0]+self.logreg_cm[1,1]),"correct predictions\033[1m")
        print("\033[1m The result is telling us that we have: ",(self.logreg_cm[0,1]+self.logreg_cm[1,0]),"incorrect predictions\033[1m")
        print("\033[1m We have a total predictions of: ",(self.logreg_cm.sum()))
        print(classification_report(self.y_test, self.logreg.predict(self.x_test)))

        print("\033[1m The result is telling us that we have: ",(self.forest_cm[0,0]+self.forest_cm[1,1]),"correct predictions\033[1m")
        print("\033[1m The result is telling us that we have: ",(self.forest_cm[0,1]+self.forest_cm[1,0]),"incorrect predictions\033[1m")
        print("\033[1m We have a total predictions of: ",(self.forest_cm.sum()))
        print(classification_report(self.y_test, self.rf.predict(self.x_test)))

        print("\033[1m The result is telling us that we have: ",(self.svc_cm[0,0]+self.svc_cm[1,1]),"correct predictions\033[1m")
        print("\033[1m The result is telling us that we have: ",(self.svc_cm[0,1]+self.svc_cm[1,0]),"incorrect predictions\033[1m")
        print("\033[1m We have a total predictions of: ",(self.svc_cm.sum()))
        print(classification_report(self.y_test, self.svc.predict(self.x_test)))

class Evaluation:
    @app.route('/')
    @app.route('/index')
    def index():
        return render_template("index.html")

    @app.route('/api/evaluation', methods=['GET'], endpoint = 'evaluation')
    def evaluation():
        try:
            Table = request.args.get('print')
            HeatMap = request.args.get('heatmap')

            if HeatMap == "HeatMap":
                context = Context(ConcreteEvaluationHeatMap())
                print("===========================================")
                print("Client: Strategy is set to Concrete Evaluation HeatMap.")
                context.logic()

            if Table == "Print":
                context = Context(ConcreteEvaluationResult())
                print("===========================================")
                print("Client: Strategy is set to Concrete Evaluation Result.")
                context.logic()

            return redirect("http://localhost:5005/", code=302)
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    PORT = os.environ.get('PORT', 5005)
    print("Port Number: ", PORT)
    app.run(debug=True, host='0.0.0.0', port=PORT)
