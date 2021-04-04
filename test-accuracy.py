from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import load_img, img_to_array

# no_of_images_to_load_in_each_folder = 500

image_arrays = list()
labels = list()
train_dir = os.path.join('./', 'demo_data')
class_list = os.listdir(train_dir)

for each_class in class_list:
    image_names = os.listdir(f"{train_dir}/{each_class}/")

    # for each_image in image_names[:no_of_images_to_load_in_each_folder]:   use this to limit the number of loaded images
    for each_image in image_names:
        image_full_path = f"{train_dir}/{each_class}/{each_image}"
        temp_image = load_img(image_full_path, target_size = (256, 256))
        image_arrays.append(img_to_array(temp_image))
        labels.append(each_class)

print("\n\n\n")
print('Number of images : ', len(image_arrays))
print('Number of labels : ', len(labels))

scaled_image_arrays = np.array(image_arrays, dtype = np.float32) / 255

label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(labels)

x_train, x_test, y_train, y_test = train_test_split(scaled_image_arrays, image_labels, test_size = 0.3, random_state = 1)

model_new = load_model("grape-vision-model-prototype.h5")

scores = model_new.evaluate(x_test, y_test)
print("\n\n\n")
print(f"Model Accuracy: {scores[1]*100}")
print("\n\n")