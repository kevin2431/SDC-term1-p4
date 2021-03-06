import numpy as np
import csv
import cv2
#将csv文件中的每一行都读入lines中
lines=[]
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for i,line in enumerate(reader):
		if i>0:
			lines.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images=[]
			measurements=[]
			for batch_sample in batch_samples:
				source_path=batch_sample[0]
				filename=source_path.split('/')[-1]
				current_path='/opt/carnd_p3/data/IMG/'+filename
				image=cv2.imread(current_path)
				image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
				images.append(image)
				#只对应图片 和方向盘转弯的数据
				measurement=float(batch_sample[3])
				measurements.append(measurement)

				steering_center=measurement
				# create adjusted steering measurements for the side camera images
				correction = 0.2 # this is a parameter to tune
				steering_left = steering_center + correction
				steering_right = steering_center - correction
				path_left=batch_sample[1]
				path_right=batch_sample[2]
				current_left='/opt/carnd_p3/data/IMG/'+path_left.split('/')[-1]
				current_right='/opt/carnd_p3/data/IMG/'+path_right.split('/')[-1]
				#增加左侧相机
				image=cv2.imread(current_left)
				image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
				images.append(image)
				measurements.append(steering_left)
				#增加右侧相机
				image=cv2.imread(current_right)
				image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
				images.append(image)
				measurements.append(steering_right)

			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(measurements)
			yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)



#利用Keras创建模型
from keras.models import Sequential
from keras.layers import Flatten,Dense,Activation,Lambda,Dropout,Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model=Sequential()
#normalization,mean-zero
model.add(Lambda(lambda x: (x / 127.5) - 1, input_shape=(160,320,3)))
#320x90x3
model.add(Cropping2D(cropping=((50,20), (0,0)),data_format="channels_last"))

#160x45x24
model.add(Conv2D(24,(5,5),strides=(2, 2), padding='same',activation='relu'))
#78x21x36
model.add(Conv2D(36,(5,5),strides=(2, 2), padding='valid',activation='relu'))
#37x9x48
model.add(Conv2D(48,(5,5),strides=(2, 2), padding='valid',activation='relu'))
#17x3x64
model.add(Conv2D(64,(5,5),strides=(2, 2), padding='valid',activation='relu'))
#15x1x64
model.add(Conv2D(64,(3,3), padding='valid',activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
#history_object=model.fit(X_train,y_train,batch_size=128, epochs=10,validation_split=0.2,shuffle=True,verbose=1)

history_object=model.fit_generator(train_generator, steps_per_epoch= len(train_samples),validation_data=validation_generator, validation_steps=len(validation_samples), epochs=2, verbose = 1)

import matplotlib.pyplot as plt

print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss.jpg')

model.save('model.h5')