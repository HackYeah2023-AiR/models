from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
from PIL import Image

test_path = 'alexnet_model/images/pets2/test_set2/'

cates = ['dogs', 'cats']
def load_images_and_labels(data_path, cates): 
  X = []
  i = 0
  for index, cate in enumerate(cates): 
    for img_name in os.listdir(data_path + cate + '/'):
      i = i +1
      img = cv2.imread(data_path + cate + '/' + img_name)
      if img is not None: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = Image.fromarray(img, 'RGB')
        img_rs = img_array.resize((227,227))
        img_rs = np.array(img_rs)
        X.append(img_rs)
  return X

X_test= load_images_and_labels(test_path, cates)

def preprocess_data(X):
  X = np.array(X)
  X = X.astype(np.float32)
  X = X/255.0
  return X
  
X_test = preprocess_data(X_test)

model = load_model("Alexnet_model.h5")

plt.figure(figsize = (15,8))
cate = ['dog', 'cat']
for i in np.arange(8):
  ind = random.randint(0, len(X_test))
  img = X_test[ind]
  img_rs = img.reshape(1,227,227,3)
  y_pred = model.predict(img_rs)
  predicted_cate = cate[np.argmax(y_pred)]
  plt.subplot(240+1+i)
  plt.imshow(img)
  plt.title('predicted: ' + str(predicted_cate))
plt.show()