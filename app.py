from flask import Flask
from logging import Logger
from flask import render_template
from model_wrapper import ModelWrapper

app = Flask(__name__)
logger = Logger()

try:
    ml_model = ModelWrapper("/Users/eugene/mlPractice/CyberAttackModel/cyberAttackModel/cyber_attack_model.ipynb")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    ml_model = None 
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

if(__name__ == '__main__'):
    app.run(debug = True)