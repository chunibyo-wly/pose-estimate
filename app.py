import cv2
import numpy
import torch
from flask import Flask, jsonify, request

from poseService import PoseService

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/getPose', methods=['PUT', 'POST'])
def get_pose():
    image = request.files['file'].read()
    image = numpy.fromstring(image, numpy.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # image = cv2.imread('person.png')
    print(image.shape, torch.cuda.is_available())
    pose = PoseService(image).get_pose()
    return jsonify(pose)


@app.route('/get', methods=['GET'])
def _get():
    return jsonify({"s": "dfdfd"})


if __name__ == '__main__':
    app.run()
