# Serve model as a flask application
import tensorflow as tf
import joblib
from flask import Flask, render_template, session, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField

rnn_twt_model = None
tokenizer = None

# load model at the beginning once only
def loadmodel():
	global rnn_twt_model
	rnn_twt_model = tf.keras.models.load_model("rnn_twt_model.h5")
	return rnn_twt_model
	
loadmodel()

# load tokenizer at the beginning once only
def load_tokenizer():
    global tokenizer
    tokenizer = joblib.load('twitter_tokenizer.pkl')
    return tokenizer
	
load_tokenizer()

# encode the raw text
def encode_text(tokenizer, raw_text):
    # encode text sequence to integer
    encoded = tokenizer.texts_to_sequences([raw_text])
    # pad encoded sequences
    encoded_text = tf.keras.preprocessing.sequence.pad_sequences(encoded, maxlen=32, padding='post')
    return encoded_text
	
def return_prediction(rnn_twt_model, tokenizer, text):
	#raw_text = str(request.data.decode("utf-8"))
	raw_text = text['twt']
	print(f'Raw Text : {raw_text}')
	encoded_text = encode_text(tokenizer, raw_text)
	print(f'Encoded Text : {encoded_text}')
	prediction = rnn_twt_model.predict(encoded_text)
	print(f'Prediction : {prediction}')
	if float(prediction[0]) >= 0.50:
		return "Positive"
	else:
		return "Negative"

app = Flask(__name__)

# Configure a secret SECRET_KEY
app.config['SECRET_KEY'] = 'twtsntmnt'

# Now create a WTForm Class http://wtforms.readthedocs.io/en/stable/fields.html
class TweetForm(FlaskForm):
    tweet = TextField('Please enter a Tweet to check its Sentiment : ')

    submit = SubmitField('Analyze')

@app.route('/', methods=['GET', 'POST'])
def index():
	# Create instance of the form.
    form = TweetForm( )
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit( ):
        # Grab the data from the breed on the form.
        session['tweet'] = form.tweet.data

        return redirect(url_for("prediction"))

    return render_template('home.html', form=form)

@app.route('/prediction')
def prediction():

    content = { }

    content['twt'] = str(session['tweet'])

    results = return_prediction(rnn_twt_model, tokenizer, text=content)

    return render_template('prediction.html',results=results)


if __name__ == '__main__':
	app.run(debug=True)