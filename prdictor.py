import  tensorflow as tf
import  numpy as np
import requests
import os
import matplotlib.pyplot as plt
import matplotlib.image as image
from flask import Flask, request, jsonify
from urllib.request import urlopen
from PIL import Image
import requests

app = Flask(__name__)


def prediction_with_model(model,image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image,channels = 3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image,[60,60])#(60,60,3)
    image = tf.expand_dims(image,axis=0)#(1,60,60,3)

    prediction = model.predict(image)
    print(prediction)

    prediction = np.argmax(prediction)
    return prediction

@app.route('/identify-picture', methods=['GET', 'POST'])
def about():
    if request.method == 'POST':
        # data = request.get_json()
        model = tf.keras.models.load_model("./modelnew2")
        response = requests.get(request.get_json()["picture"])
        # response = requests.get('https://5.imimg.com/data5/NS/VE/MY-38748036/designer-midi-top-500x500.jpg')
        with open('image.jpg', 'wb') as f:
            f.write(response.content)
            f.raw.decode_content = True
        prediction = prediction_with_model(model, os.path.abspath(os.getcwd()) + '/image.jpg')
        # imgplot = plt.imshow(os.path.abspath(os.getcwd()) + '/image.jpg')
        if prediction == 0:
            return jsonify({
                "pictureLink": request.get_json()["picture"],
                "pictureType": "casual"
                })
        elif prediction == 1:
            return jsonify({
                "pictureLink": request.get_json()["picture"],
                "pictureType": "offical"
                })
        elif prediction == 2:
            return jsonify({
                "pictureLink": request.get_json()["picture"],
                "pictureType": "sports"
                })
        else:
            return jsonify({
                "pictureLink": request.get_json()["picture"],
                "pictureType": "unknown"
                })
    return 'This is a get request'


# if __name__ == "__main__":
#     app.run(debug=True)



