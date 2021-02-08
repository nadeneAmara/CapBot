from flask import Flask, render_template
from model.py import *

app = Flask(__name__)

@app.route("/")
def index():
    pass