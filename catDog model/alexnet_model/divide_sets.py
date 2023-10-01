
import os
import random
import shutil

data_path = "model/hack/alexnet_model/images/pets2/Dog"

# path to destination folders
train_folder = os.path.join(data_path, '../training_set2/dogs')
test_folder = os.path.join(data_path, '../test_set2/dogs')

# Define a list of image extensions
image_extensions = ['.jpg']

# Create a list of image filenames in 'data_path'
imgs_list = [filename for filename in os.listdir(data_path) if os.path.splitext(filename)[-1] in image_extensions]

# Sets the random seed 
random.seed(42)

# Shuffle the list of image filenames
random.shuffle(imgs_list)

# determine the number of images for each set
train_size = int(len(imgs_list) * 0.95)
test_size = int(len(imgs_list) * 0.05)

# Create destination folders if they don't exist
#for folder_path in [train_folder, test_folder]:
for folder_path in [ test_folder]:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Copy image files to destination folders
for i, f in enumerate(imgs_list):
 #   if i < train_size:
 #       dest_folder = train_folder
 #   else:
    
    if i > train_size:
        dest_folder = test_folder
    else:
        dest_folder = ''
    if dest_folder != '':
        shutil.copy(os.path.join(data_path, f), os.path.join(dest_folder, f))