from flask import Flask, render_template, request
import sentiment as senti

app= Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/sub',methods=['POST'])
def senti_ana():
    if request.method == 'POST':
        data= request.form['data']
        data_pred= senti.model.predict([data])
       
    return render_template("sub.html", dp=data_pred,d=data)



if __name__ == '__main__':
    app.run(debug=True)