import os
import shutil
import operator

DATA_PATH = os.path.join('..', 'data')
DATASET_PATH = os.path.join('..', 'data', '3D-FUTURE-model')

if __name__ == '__main__':
    if not os.path.exists(DATASET_PATH):
        print('DATASET NOT EXIST.')
        exit()
    CACHE_PATH = os.path.join(DATA_PATH, '_cache')
    if not os.path.exists(CACHE_PATH):
        os.mkdir(CACHE_PATH)
    shutil.copyfile(os.path.join(DATASET_PATH, 'categories.py'), os.path.join(CACHE_PATH, 'categories.py'))