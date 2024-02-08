import numpy as np
import os
from keras.preprocessing import image
import keras.utils as image
from keras.models import load_model

model = load_model("best_model.hdf5")


def predict_image(image_path):
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    print(result, type(result))
    res = int(result[0][0])
    print(res, type(res))
    if res == 1:
        prediction = 'E'
    else:
        prediction = 'F'
    print("predicted class", prediction)
    return res

model = load_model('best_model.hdf5')
'''
img_path='dataset/test/E/E5.jpg'
pred_result=predict_image(img_path)
print(pred_result)
'''
predicted_list = []
original_list = []
for img_fl in os.listdir('./dataset/test/E'):
    img_path = os.path.join('dataset/test/E', img_fl)
    pred_result = predict_image(img_path)
    print(img_path, '.....', pred_result)
    predicted_list.append(pred_result)
    original_list.append(0)
print(11111, predicted_list)
print(2222, original_list)
