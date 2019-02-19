# USAGE
# python classify.py --model final.model --labelbin final.pickle --image ./examples/crime_madong.jpg
# python classify.py --model pokedex.model --labelbin lb.pickle --image ./examples/pikachu_toy.png
# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import matplotlib.pyplot as plt
from keras.models import Model
from sklearn import preprocessing

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

def detect_faces(image_path):
    image_grey = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    faces=FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)
    print(faces)
    sub_img = []
    for x,y,w,h in faces:
        sub_img=image_grey[y-10:y+h+10,x-10:x+w+10]
    return sub_img

new_path='/home/minkj1992/anaconda3/envs/opencv/share/OpenCV/haarcascades/'
FACE_CASCADE = cv2.CascadeClassifier(new_path+'haarcascade_frontalface_default.xml')

# load the image and grayScale
image = detect_faces(args["image"])

# # image = cv2.imread(args["image"])
# output = image.copy()

# # pre-process the image for classification
# # 512 * 512
# image = cv2.resize(image, (96, 96))
# image = image.astype("float") / 255.0
# image = img_to_array(image)
# image = np.expand_dims(image, axis=0)


# # load the trained convolutional neural network and the label
# # binarizer
# print("[INFO] loading network...")
# model = load_model(args["model"])

# # pickle.dump(lb,open(args["labelbin"], "rb").read(),protocol=2)
# lb = pickle.loads(open(args["labelbin"], "rb").read())

# # show stage upload value (custom)
# layer_name='dense_2'
# intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
# intermediate_output = intermediate_layer_model.predict(image)


# # classify the input image
# print("[INFO] classifying image...")
# proba = model.predict(image)[0]

# idx = np.argmax(proba)
# label = lb.classes_[idx]

# # we'll mark our prediction as "correct" of the input image filename
# # contains the predicted label text (obviously this makes the
# # assumption that you have named your testing image files this way)
# filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
# correct = "correct" if filename.rfind(label) != -1 else "incorrect"

# # build the label and draw the label on the image
# label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
# output = imutils.resize(output, width=400)
# cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
# 	0.7, (0, 255, 0), 2)


# # show probability(minkj1992)
# # label_binarizer.inverse_transform([y_train[0]])
# for i,j in zip(list(lb.classes_),list(*intermediate_output)):
#     print("{}: {:.2f}%".format(i,j*100))


# # show the output image
# print("[INFO] {}".format(label))
# plt.imshow(output)

# # cv2.imshow("Output", output)
# plt.show()
# # cv2.waitKey(0)
