from flask import Flask,render_template,url_for,request
import pickle
import joblib

filename = 'pickle.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('Home.html')

@app.route('/Send.html')
def send():
	return render_template('Send.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('Send.html',prediction = my_prediction)

@app.route('/Box.html')
def box():
	return render_template('Box.html')

if __name__ == '__main__':
	app.run(debug=True)
