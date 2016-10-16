from flask import Flask, render_template, request, jsonify
import numpy
from scikitlearnMNIST import *

app = Flask(__name__)

@app.route("/")
def mnist():
    return render_template("mnist.html")

@app.route("/estimate", methods = ["POST"])
def estimate():
    try:
        x = numpy.array([request.json["input"]]) / 255.0
        #print(x.shape, x[0, 300:400]*255)
        y = int(numberpredict(x))
        return jsonify({"estimated":y} )
    except Exception as e:
        print(e)
        return jsonify({"error":e})



if __name__ == "__main__":
    app.run()

