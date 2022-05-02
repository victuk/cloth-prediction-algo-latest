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
        response = requests.get(request.form.get('picture'))
        # response = requests.get('https://5.imimg.com/data5/NS/VE/MY-38748036/designer-midi-top-500x500.jpg')
        with open('image.jpg', 'wb') as f:
            f.write(response.content)
            f.raw.decode_content = True
        prediction = prediction_with_model(model, os.path.abspath(os.getcwd()) + '/image.jpg')
        # imgplot = plt.imshow(os.path.abspath(os.getcwd()) + '/image.jpg')
        if prediction == 0:
            return jsonify({
                "pictureLink": request.form.get('picture'),
                "pictureType": "casual"
                })
        elif prediction == 1:
            return jsonify({
                "pictureLink": request.form.get('picture'),
                "pictureType": "offical"
                })
        elif prediction == 2:
            return jsonify({
                "pictureLink": request.form.get('picture'),
                "pictureType": "sports"
                })
        else:
            return jsonify({
                "pictureLink": request.form.get('picture'),
                "pictureType": "unknown"
                })
    return 'This is a get request'


if __name__ == "__main__":
    # r = requests.get('https://5.imimg.com/data5/NS/VE/MY-38748036/designer-midi-top-500x500.jpg')
    # r = open("sample_image.png", "wb")
    # r.write(r.content)
    # r.close()
    # f = Image.open(r)
    # if r.status_code == 200:
        # with open(path, 'wb') as f:
            # f.write(r.content)
            # r.raw.decode_content = True
            # shutil.copyfileobj(r.raw, f)
        
    # http = urllib3.PoolManager()
    # f = http.request('GET', 'https://image.shutterstock.com/image-photo/fulllength-portrait-cute-cheerful-smiling-260nw-615099686.jpg')
    # http = urllib3.PoolManager()
    # r = http.request('GET', 'http://www.solarspace.co.uk/PlanetPics/Neptune/NeptuneAlt1.jpg')
    # resized_image = Image.open(StringIO(r.data))
    # img = image.imread(r)
    # imgplot = plt.imshow(img)
    # plt.show()

    app.run(debug=True)

    #image_path1 ="C:\\Users\\Daniel Samuel\\Desktop\\cloth_recommendation\\fashion clothe\\myntradataset\\images\\12083.jpg"
    # image_path2 = "C:\\Users\\Daniel Samuel\\Desktop\\cloth_recommendation\\dsbs'\\0c224954-0e0f-4caa-82c8-cf9581e89336.jpg"
    # image_path3 = "C:\\Users\\Daniel Samuel\\Desktop\\cloth_recommendation\\dsbs'\\1a08f33a-2ff4-4fb8-920b-8ff514bcdda8.jpg"
    # image_path4 = "C:\\Users\\Daniel Samuel\\Downloads\\img\\img\\Abstract_Print_Colorblock_Top\\img_00000023.jpg"
    # image_path5 = "C:\\Users\\Daniel Samuel\\Desktop\\cloth_recommendation\\train\\2\\9533.jpg"
    # image_path6 = "C:\\Users\\Daniel Samuel\\Desktop\\cloth_recommendation\\train\\0\\12089.jpg"
    # image_path7 = "C:\\Users\\Daniel Samuel\\Desktop\\cloth_recommendation\\train\\0\\30886.jpg"
    # image_path8 = "C:\\Users\\Daniel Samuel\\Downloads\\pngegg.png"
    #
    # image_path = image_path4
    #
    # model = tf.keras.models.load_model("./model")
    # prediction = prediction_with_model(model,image_path)
    #
    # img = image.imread(image_path)
    #
    # imgplot = plt.imshow(img)
    # print(prediction)
    #
    # if prediction ==0 :
    #     plt.title("this is a casual cloth")
    # elif prediction == 1:
    #     plt.title("this is a offical cloth")
    # elif prediction == 2:
    #     plt.title("this is a sport  cloth")
    # else:
    #     plt.title("unknown")



