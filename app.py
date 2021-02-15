from flask import Flask, render_template, jsonify
from flask import request as flask_request
from model import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate',methods=['POST'])
def generate():
    url = flask_request.form['url']
    caption = main(url)
    return render_template('index.html', caption_text = '{}'.format(caption))

if __name__ == "__main__":
    app.run(debug=True)
