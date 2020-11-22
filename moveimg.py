import os
import shutil

path = os.getcwd() + '/dataset'
new_path = os.getcwd() + '/train_dir/test_img'

for root, dirs, files in os.walk(path):
    for i in range(len(files)):
        if (files[i][-3:] == 'jpg'):
            file_path = root + '/' + files[i]
            new_file_path = new_path + '/' + files[i]
            shutil.copy(file_path, new_file_path)
