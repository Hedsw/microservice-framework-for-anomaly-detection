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

from flask import redirect

import os

from flask import Flask
from flask import request, jsonify, render_template, redirect
import sys

app=Flask(__name__, template_folder='../templates')


class Evaluation:
    @app.route('/')
    @app.route('/index')
    def index():
        return render_template("index.html")

    @app.route('/api/evaluation', methods=['GET'], endpoint = 'evaluation')
    def evaluation():
        try:
            df = pd.read_csv('../dataset/process_data.csv')
            y = df['fraud']
            X =  df.loc[:, df.columns != 'fraud']
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

            # Logistic Regression
            print ("xxxxxxxxxxxxxxx Logistic Regression xxxxxxxxxxxxxxx")
            logreg = LogisticRegression(random_state=42)
            logreg.fit(x_train, y_train)
            logreg_y_pred = logreg.predict(x_test)
            plt.clf()
            matplotlib.rcParams["figure.figsize"] = (10.0, 6.0)
            logreg_cm = confusion_matrix(logreg_y_pred, y_test, labels=[1,0])
            sns.heatmap(logreg_cm, cmap="RdPu", annot=True, fmt=".0f",xticklabels = ["Fraudulent", "Legitimate"], yticklabels = ["Fraudulent", "Legitimate"])
            plt.ylabel("True class")
            plt.xlabel("Predicted class")
            plt.title("Logistic Regression")
            plt.savefig("logistic_regression")
            print("\033[1m The result is telling us that we have: ",(logreg_cm[0,0]+logreg_cm[1,1]),"correct predictions\033[1m")
            print("\033[1m The result is telling us that we have: ",(logreg_cm[0,1]+logreg_cm[1,0]),"incorrect predictions\033[1m")
            print("\033[1m We have a total predictions of: ",(logreg_cm.sum()))
            print(classification_report(y_test, logreg.predict(x_test)))

            # Random Forest
            print ("xxxxxxxxxxxxxxx Random Forest xxxxxxxxxxxxxxx")
            plt.clf()
            rf = RandomForestClassifier(random_state=42)
            rf.fit(x_train, y_train)
            y_pred = rf.predict(x_test)
            forest_cm = confusion_matrix(y_pred, y_test, labels=[1,0])
            sns.heatmap(forest_cm, cmap="RdPu", annot=True, fmt=".0f",xticklabels = ["Fraudulent", "Legitimate"], yticklabels = ["Fraudulent", "Legitimate"])
            plt.ylabel("True class")
            plt.xlabel("Predicted class")
            plt.title("Random Forest")
            plt.savefig("random_forest")
            print("\033[1m The result is telling us that we have: ",(forest_cm[0,0]+forest_cm[1,1]),"correct predictions\033[1m")
            print("\033[1m The result is telling us that we have: ",(forest_cm[0,1]+forest_cm[1,0]),"incorrect predictions\033[1m")
            print("\033[1m We have a total predictions of: ",(forest_cm.sum()))
            print(classification_report(y_test, rf.predict(x_test)))

            # Support Vector Machine
            print ("xxxxxxxxxxxxxxx Support Vector Machine xxxxxxxxxxxxxxx")
            plt.clf()
            svc = SVC(random_state=42)
            svc.fit(x_train, y_train)
            svc_y_pred = svc.predict(x_test)
            svc_cm = confusion_matrix(svc_y_pred, y_test, labels=[1,0])
            sns.heatmap(svc_cm, cmap="RdPu", annot=True, fmt=".0f",xticklabels = ["Fraudulent", "Legitimate"], yticklabels = ["Fraudulent", "Legitimate"])
            plt.ylabel("True class")
            plt.xlabel("Predicted class")
            plt.title("Support Vector Machine")
            plt.savefig("support_vector_machine")
            print("\033[1m The result is telling us that we have: ",(svc_cm[0,0]+svc_cm[1,1]),"correct predictions\033[1m")
            print("\033[1m The result is telling us that we have: ",(svc_cm[0,1]+svc_cm[1,0]),"incorrect predictions\033[1m")
            print("\033[1m We have a total predictions of: ",(svc_cm.sum()))
            print(classification_report(y_test, svc.predict(x_test)))

            # return jsonify({'sucess': 200, "message":"preprocessing is task is done"})
            return redirect("http://localhost:5005/", code=302)
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    PORT = os.environ.get('PORT', 5005)
    print("Port Number: ", PORT)
    app.run(debug=True, host='0.0.0.0', port=PORT)
