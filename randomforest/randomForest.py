import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection

import os

from flask import Flask
from flask import request, jsonify, render_template, redirect
import sys

app=Flask(__name__, template_folder='../templates')

class Randomforest:
    @app.route('/')
    @app.route('/index')
    def index():
        return render_template("index.html")

    @app.route('/api/randomforest', methods=['GET'], endpoint = 'randomforest')
    def randomforest():
        try:
            # Service 2 
            df = pd.read_csv('../dataset/process_data.csv')
            print(df)
            y = df['fraud']
            X =  df.loc[:, df.columns != 'fraud']
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            
            """
            Different from Here 
            """
            logreg = LogisticRegression(random_state=42)
            logreg.fit(x_train, y_train)
            print("===========================================")
            print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(y_test, logreg.predict(x_test))))
            kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle = True)
            print("===========================================")
            print("Kfold is ready")
            modelcv = RandomForestClassifier(random_state=42)
            print("===========================================")
            print("Classifier is ready")
            scoring = 'accuracy'
            results = model_selection.cross_val_score(modelcv, x_train, y_train, cv = kfold, scoring = scoring)
            print("===========================================")
            print("10-fold cross validation average accuracy of the random forest model: %.3f" % (results.mean()))
            return jsonify({'sucess': 200, "message":"randomforest is task is done", "cross validation average accuracy": results.mean()})
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})
    
        
        

if __name__ == '__main__':
    PORT = os.environ.get('PORT', 5002)
    print("Port Number: ", PORT)
    app.run(debug=True, host='0.0.0.0', port=PORT)
