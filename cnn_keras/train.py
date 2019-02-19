# USAGE
# python train.py --dataset dataset --model final3.model --labelbin final3.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# python argparse를 활용한 읽어야할 데이터 루트 설정
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# 학습횟수 및 이미지 dimension 조정
EPOCHS = 50
INIT_LR = 1e-3
BS = 32
# gray_scale
IMAGE_DIMS = (96, 96, 1)
# rgb color
# IMAGE_DIMS = (96, 96, 3)

# x: 데이터, y: labels
data = []
labels = []

# 이미지 읽어들인 뒤, random하게 train데이터를 섞어준다.
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

#  loop를 돌면서 이미지들을 업로드
for imagePath in imagePaths:
	# resize, 이미지 array화
	image = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	# 데이터 list에 image 부착
	data.append(image)
 
	# 이미지 파일 이름을 dataset/{CLASS_LABEL}/{FILENAME}.jpg로 저장하여 label을 바로 읽어들이게 한다.
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)


# scale
# max value로 scale화 시켜주어 range [0,1]로 데이터가 저장되도록 한다.
# 계산 복잡도 낮춰준다.
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))

# multiple classification이므로 label을 차원화 시켜준다.(binarization) 
# ex 첫번째 라벨 [1,0,0,0,0,0]
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# print(list(lb.inverse_transform(labels)))
# exit()

# 80% train data, 20% valid data
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

# 데이터 수가 적으므로 ImageGen을 사용하여 각도에 따라서 데이터를 조금씩 변형시켜 train데이터의 수를 늘려준다.
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# smallervggnet class를 불어와 이미지 모델을 init시켜준다.
print("[INFO] compiling model...")
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(lb.classes_))
#  Adam 을 통하여 학습을 진행한다.(gradient descent)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# 정확도 측정 categorical_crossentropy
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# print summary
print(model.summary())

# train 
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save , 직렬화 시켜서 저장시킨다.
print("[INFO] serializing network...")
model.save(args["model"])

# save, pickle화 시켜서 모델을 저장한다.
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

# 학습 loss와 accu및 hyperparameter들을 보여준다.
# 학습 진행사항 show
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])


