from flask import render_template

from flask import Flask

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')