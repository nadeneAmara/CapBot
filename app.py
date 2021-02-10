from flask import Flask, render_template, jsonify, request
from model.py import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    #caption = main()

@app.route('/generate',methods=['POST'])
def generate():
    url = request.form['url']
    caption = main(url)
    return render_template('index.html', caption_text = '$ {}'.format(caption))
