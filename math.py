# 수학 기호 인식 모델

# Setting
import sys
import os
sys.path.append(os.getcwd())
from config import *
from models import *

# 경로 추가

# Data Load
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#train_generator = train_datagen.flow_from_directory('data/train', target_size=(64, 64), batch_size=128, class_mode='categorical')
#val_generator = val_datagen.flow_from_directory('data/val', target_size=(64, 64), batch_size=128, class_mode='categorical')

train_generator = train_datagen.flow_from_directory('C:/Users/YY/Dropbox/data/train', target_size=(64, 64), batch_size=128, class_mode='categorical')
val_generator = val_datagen.flow_from_directory('C:/Users/YY/Dropbox/data/val', target_size=(64, 64), batch_size=128, class_mode='categorical')
test_generator = test_datagen.flow_from_directory('C:/Users/YY/Dropbox/data/test', target_size=(64, 64), batch_size=1, class_mode='categorical')


# Class 확인
print('Your Class: ', '\n', val_generator.class_indices)

# Model Define
# K.set_learning_phase(1)
K.set_image_data_format('channels_last')

# 학습
# C = Config()
model = ResNet50(input_shape=(64, 64, 3), classes=29)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

history = model.fit_generator(train_generator, steps_per_epoch=353, epochs=1,
                              validation_data=val_generator, validation_steps=3, callbacks=[early_stop])

# 학습과정 살펴보기
show_loss_graph(history=history)

# Weight 저장
# model.save_weights('weights/WEIGHTS.h5')
model.load_weights('weights/WEIGHTS3.h5')

"""
img = image.load_img('C:/Users/YY/Dropbox/data/test/s.jpg', target_size=(64, 64))
img_input = image.img_to_array(img)/255.
INPUT = K.expand_dims(img_input, axis=0)
model.predict(INPUT, steps=1)

get_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[176].output])
# test mode = 0
output = get_output([INPUT, 1])[0]
np.argsort(output[0])[::-1][0:5]
"""


# 테스트 결과 출력
output = model.predict_generator(test_generator, verbose=0)
dic = test_generator.class_indices
idx = np.argsort(output, axis=1)[0][::-1][0]
print("Test 이미지는", list(dic.keys())[idx], "입니다")














