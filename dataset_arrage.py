import csv
import os
import glob
from sklearn.model_selection import train_test_split
import  shutil
import matplotlib.pyplot as plt
import  numpy as np

def display_images(example,labels):
   plt.figure(figsize=(10,10))
   for i in range(25):
      idx = np.random.randint(0,example.shape[0]-1)
      img = example[idx]
      label = labels[idx]
      plt.subplot(5,5,i+1)
      plt.title(str(label))
      plt.tight_layout()
      plt.imshow(img,cmap='gray')
   plt.show()
path_to_save = "C:\\Users\\Daniel Samuel\\Downloads\\img"


def split_data(path_to_data):
    main_folders = os.listdir(path_to_data)
    for x in  main_folders:
            images_path = glob.glob(os.path.join(x,"*.jpg"))
         #   img_full_path = os.path.join(x, images_path)
            shutil.move(images_path, path_to_save)

      #  shutil.copy(images_path, path_to_save)
    #    x_train,x_val = train_test_split(images_path,test_size=split_Size)
    #     for x in x_train:
    #         #basename = os.path.basename(x)  ###   get the name of the image
    #         path_to_folder = os.path.join(path_to_Save_train,each_folder)
    #         if not os.path.isdir(path_to_folder):
    #             os.makedirs(path_to_folder)
    #         shutil.copy(x,path_to_folder)
    #     for x in x_val:
    #         #basename = os.path.basename(x)  ###   get the name of the image
    #         path_to_folder = os.path.join(path_to_save_val,each_folder)
    #         if not os.path.isdir(path_to_folder):
    #             os.makedirs(path_to_folder)
    #         shutil.copy(x,path_to_folder)

path_to_data = "C:\\Users\\Daniel Samuel\\Downloads\\img\\img"


split_data(path_to_data)
