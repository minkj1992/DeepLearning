[참고자료](https://subinium.github.io/Keras-5-2/)
# 사전 훈련된 ConvNet 사용하기
- 특성 추출
- 미세 조정

## 특성 추출 (Feature Extraction)

convnet은 이미지 분류를 위해 두 part로 구성된다.
- 합성곱 기반 층(합성곱과 폴링 layer)
- Fully connected Classifier

특정 합성곱 층에서 추출한 표현의 일반성 및 재사용성 수준은 모델에 있는 층의 깊이에 따라 달려있습니다.

- 하위 층 : 에지, 색깔, 질감 등 지역적이고 매우 일반적인 특성 맵
- 상위 층 : 강아지 눈, 고양이 귀와 같은 좀 더 추상적인 개념

새로운 데이터셋과 훈련된 데이터셋이 많이 다르다면 전체 합성곱 기반 층이 아닌 모델의 하위 층 몇 개만 특성 추출에 사용하는 것이 좋습니다.

ImageNet 클래스에서는 강아지와 고양이가 있기 때문에 완전 연결 층 정보를 재사용해도 좋지만 일반적인 케이스를 위해 여기서는 사용하지 않습니다. ImageNet 데이터셋에서 훈련된 VCG16 네트워크에서 유용한 특성을 추출합니다. 그 후 특성으로 훈련을 진행합니다.


```python
# 코드 5-16 VGG16 합성곱 기반 층 만들기
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
```

- @데이터 증식이란?

> 추출된 특성의 크기는 (samples, 4, 4, 512)이기에 완전 연결 분류기에 넣기 위해 학습시킬 데이터의 shape를 (samples, 8192)크기로 펼칩니다.

```python
#코드 5-17 사전 훈련된 합성곱 기반 층을 사용한 특성 추출하기 + 특성맵 펼치기
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = './datasets/학습시킬 폴더'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # 제너레이터는 루프 안에서 무한하게 데이터를 만들어내므로 모든 이미지를 한 번씩 처리하고 나면 중지합니다
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))
```

이후 모델을 정의 해준다.

```python
# 코드 5-18 완전 연결 분류기를 정의하고 훈련하기
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))
```

## 클래스 활성화의 히트맵 시각화하기

`클래스 활성화 맵(Class Activation Map, CAM)`

1. binary classifier를 활용한 
