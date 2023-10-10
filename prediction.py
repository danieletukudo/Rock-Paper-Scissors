import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import matplotlib.image as image
# Building the AI model



model  = tf.keras.models.load_model('model.h5')

#
# images  =os.listdir('test')
# for i in images:
#
#     path = 'test/' + i
#     print(path)
#
#     img = load_img(path,target_size = (150,150))
#
#     x = img_to_array(img)
#     x /=255
#
#     x = np.expand_dims(x,axis=0)
#
#
#     images = np.vstack([x])
#     prediction = model.predict(images,batch_size=10)
#     prediction = np.argmax(prediction)
#     #
#     test_image = image.imread(path)
#     imgplot = plt.imshow(test_image)
#     print(prediction)
#
#     if prediction == 0:
#         plt.title("Papper")
#         plt.figure()
#
#     elif prediction == 1:
#         plt.title("Rock")
#         plt.figure()
#
#     elif prediction == 2:
#         plt.title("Scissors")
#         plt.figure()
#
# # plt.show()
#
#
#
#
#
#
#
path = "paper01-000.png"
#
# # print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
# PLOTTING THE MODEL ACCURACY AND LOSS



img = load_img(path,target_size = (150,150))

x = img_to_array(img)
x /=255

x = np.expand_dims(x,axis=0)

images = np.vstack([x])
prediction = model.predict(images,batch_size=10)

prediction = np.argmax(prediction)
print(prediction)


test_image = image.imread(path)
imgplot = plt.imshow(test_image)

if prediction  == 0:
            plt.title("Paper")

elif prediction == 1 :
            plt.title("Rock")

elif prediction == 2 :
            plt.title("Scissors")
plt.show()


