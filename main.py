from flask import Flask,render_template,request
import sys, subprocess
sys.path.append(".")
from src.pipeline.training_pipeline import train_model
from src.pipeline.prediction_pipeline import webcamdetection

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train",methods=["POST"])
def training():
    if request.method=="POST":
        train_model()
    return render_template("training.html")

@app.route("/test",methods=["POST"])
def predicting():
    if request.method == 'POST':
        try:
            # Run your webcam.py script as a subprocess
            subprocess.Popen(['python', 'webcam.py'])
            return "<h1> A webcam will appear and predict the sign. Press 'q' to quit</h1>" 
        except Exception as e:
            return f"Error: {str(e)}"

if __name__=="__main__":
    app.run(debug=True)