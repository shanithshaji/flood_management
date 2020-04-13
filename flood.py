import flask
import warnings
import pandas as pd
from flask import Flask,request,render_template,jsonify,Response
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report

app = Flask(__name__)
@app.route('/')
def index():
    s="  "
    return render_template('index.html')
@app.route('/form')
def re1():
      return render_template('form.html')

@app.route("/formaction",methods=['GET','POST'])
def action():
    if request.method=='POST':
        tem= request.form['temperature']
        hum =request.form['humidity']
        alti =request.form['altitude']
        rain = request.form['rain']
        data=pd.read_csv('static/complete_data.csv')
        df_flood = data
        X=df_flood.iloc[:,1:5]
        X = preprocessing.normalize(X)
        Y=df_flood.iloc[:,5:6]
        a = pd.DataFrame(Y)
        b = a.as_matrix().reshape(-1,1)
        a = pd.DataFrame(b)
        Y= preprocessing.normalize(a)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4,random_state=54)
        clf= RandomForestClassifier(n_estimators=100)
        clf.fit(X_train,y_train)
        X1= preprocessing.normalize([[float(tem),float(hum),float(alti),float(rain)]])
        y_pred=clf.predict(X1)
        a=int(y_pred[0])
        if a==1:
            s='FLOOD'
        else:
            s='NO-FLOOD'
        return render_template('result.html',flood=s)

if __name__ == '__main__':
    app.run(debug=True)
