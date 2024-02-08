import os
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import keras.utils as image
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, class_names,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=None):
    """function to draw confusion matrix"""
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for e, f in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(f, e, "{:0.4f}".format(cm[e,f]),
                     horizontalalignment="center",
                     color="white" if cm[e, f] > thresh else "red")
        else:
            plt.text(f, e, "{:,}".format(cm[e, f]),
                     horizontalalignment="center",
                     color="white" if cm[e, f] > thresh else "red")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
def predict_image(image_path):
    """function to predict the test images"""
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    # print(result, type(result))
    res = int(result[0][0])
    if res == 1:
        prediction = 'F'
    else:
        prediction = 'E'
    # print("predicted class....", prediction)
    return res


model = load_model('./best_model.hdf5')
predicted_list = []
original_list = []

# to predict I images
for img_fl in os.listdir('./dataset/test/E'): # provide your test class path
    img_path = os.path.join('./dataset/test/E', img_fl)
    pred_result = predict_image(img_path)
    # print(img_path, '...........', pred_result)
    predicted_list.append(pred_result)
    original_list.append(0)

for img_fl in os.listdir('./dataset/test/F'): # provide your test class path
    img_path = os.path.join('./dataset/test/F', img_fl)
    pred_result = predict_image(img_path)
    # print(img_path, '...........', pred_result)
    predicted_list.append(pred_result)
    original_list.append(1)

# plotting confusion matrix
cm = confusion_matrix(y_true=original_list, y_pred=predicted_list)
print("confusion matrix....", cm)
cm_plot_labels = ['E', 'F']  # give your corresponding class names
plot_confusion_matrix(cm, cm_plot_labels, title='confusion_matrix')
plt.savefig('cm-binary.pdf')
plt.show()


