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

train_generator = train_datagen.flow_from_directory('data/train', target_size=(64, 64), batch_size=128, class_mode='categorical')
val_generator = val_datagen.flow_from_directory('data/val', target_size=(64, 64), batch_size=128, class_mode='categorical')


# Model Define
# K.set_learning_phase(1)
K.set_image_data_format('channels_last')

# 학습
# C = Config()
model = ResNet50(input_shape=(64, 64, 3), classes=29)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
history = model.fit_generator(train_generator, steps_per_epoch=353, epochs=1, validation_data=val_generator, validation_steps=3, callbacks=[early_stop])

# 학습과정 살펴보기
show_loss_graph(history=history)


