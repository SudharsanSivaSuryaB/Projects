from flask import Flask,render_template,request,send_file
from wtforms import FileField,SubmitField
from flask_wtf import FlaskForm
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding,Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import mpld3





app = Flask(__name__)

app.config["SECRET_KEY"] = "supersecretkey"
app.config["UPLOAD_FOLDER"] = "static/testfiles"

class UploadFileFoem(FlaskForm):
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Upload file")

@app.route("/")
def home():
    return render_template("login.html")


@app.route("/sales")
def sales():
    form = UploadFileFoem()
    return render_template("salestest.html",form = form)


@app.route("/graph",methods=["POST"])
def data():
    try:
        form = UploadFileFoem()
        if request.method == "POST" and form.validate_on_submit() :
            peridiocity = request.form.get("format")
            clnm = request.form.get("text")
            # fdate = request.form.get("fromdate")
            # tdate = request.form.get("todate")
            cnt = int(request.form.get("coun"))
            print(peridiocity,clnm,cnt)
            print(type(cnt))
            file = form.file.data # First grab the file
            file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename("data.csv")))

            df = pd.read_csv("static/testfiles/data.csv")
            # df.head()

            df = df[clnm]

            err=df.isnull().sum()
            print(err)

            def NullFill(err):
                if err>0:
                    df.fillna(method="ffill",inplace=True)
            NullFill(err)
            df = df[:10000]
            # print(df)

            scaler = MinMaxScaler(feature_range=(0,1))

            df1 = scaler.fit_transform(np.array(df).reshape(-1,1))

            def dataGenerator(data : np.ndarray,n_features:int):
                i = 0
                data = data.reshape(-1)
                columns = [f"f{i+1}" for i in range(n_features)]
                columns.append("Rate")
                
                df = pd.DataFrame(columns = columns)
                while(i < len(data) - n_features):
                    features = data[i:i+n_features+1]

                    df.loc[len(df.index)] = features.tolist()
                    i+=1
                
                return df

            df1 = dataGenerator(df1,100)
            x=df1.drop('Rate',axis=1)
            # print(x)
            y = df1["Rate"]

            callback = EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            )

            model = Sequential()

            model.add(LSTM(50,return_sequences=True,input_shape = (100,1)))
            model.add(LSTM(50,return_sequences=True))
            model.add(LSTM(50))
            model.add(Dense(1))

            model.compile(loss="mean_squared_error",optimizer="adam")
            model.summary()

            try:
                history = model.fit(x,y,epochs=1,batch_size=64)
            except KeyboardInterrupt:
                del model

            model.save("model.h5")

            pred = []
            actual = []

            pred.extend(scaler.inverse_transform(np.array(model.predict(x)).reshape(-1,1)).tolist())

            actual.extend(scaler.inverse_transform(np.array(y).reshape(-1,1)).tolist())

            print(model.predict(x))
            print(pred[:5])
            print(actual[:5])

            n = 0
            if (peridiocity == "Weekly"):
                n = 7
            elif(peridiocity == "Month"):
                n = 30
            else:
                n = 365
            def future_data(data: np.ndarray, model: tf.keras.models.Sequential, n_features:int , n_days:int,n:int):


                data = data.reshape(-1)
                

                columns = [f"f{i+1}" for i in range(n_features)]

                temp_df = pd.DataFrame(columns=columns)

                result = []


                current_day = 0
                

                # while(current_day < n_days):

                #     feature = data[-100:]
                    
                #     temp_df.loc[len(temp_df)] = feature.tolist()

                #     res = model.predict(temp_df,verbose=0)

                #     data = np.append(data,res[0][-1])

                #     current_day+=1

                for i in range(1,n_days,n):

                    feature = data[-100:]
                    
                    temp_df.loc[len(temp_df)] = feature.tolist()

                    res = model.predict(temp_df,verbose=0)

                    data = np.append(data,res[0][-1])

                    
                result = scaler.inverse_transform(np.array(res).reshape(-1,1))


                return result
            
            df2 = scaler.transform(np.array(df).reshape(-1,1))
            # cnt = 8
            n_days = n*cnt
            n_features = 100
            nxt_30 = future_data(data=df2, model=model, n_features=n_features, n_days=n_days,n=n)
            
            print(len(nxt_30))

            new_df = pd.DataFrame(nxt_30,columns=["Predicted"])
            new_df.to_csv("predicted.csv")
            x = new_df.index
            y = new_df["Predicted"]
            fig , ax = plt.subplots()
            ax.bar(x,y)
            
            fig.savefig("static/graph.png")
            return render_template("graph.html",clnm=clnm)
    except Exception as e:
            return e
    
@app.route("/download")
def download_file():
    path = "predicted.csv"
    return send_file(path,as_attachment=True)
    
if __name__ == "__main__":
    app.run(debug=True)