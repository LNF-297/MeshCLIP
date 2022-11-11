import os
import shutil
import operator

DATA_PATH = os.path.join('..', 'data')
DATASET_PATH = os.path.join('..', 'data', '3D-FUTURE-model')
SCRIPT_PATH = os.path.join('.', '3d_future.mlx')
def convert(IN_PATH, OUT_PATH):
    os.system('meshlabserver -i ' + IN_PATH + ' -o ' + OUT_PATH + ' -s ' + SCRIPT_PATH)
def normalize():
    for dirs in os.listdir(DATASET_PATH):
        if operator.contains(dirs, '.'):
            continue
        IN_PATH = os.path.join(DATASET_PATH, dirs, 'raw_model.obj')
        OUT_PATH = os.path.join(DATASET_PATH, dirs, 're_normalized_model.obj')
        convert(IN_PATH, OUT_PATH)
if __name__ == '__main__':
    if not os.path.exists(DATASET_PATH):
        print('DATASET NOT EXIST.')
        exit()
    CACHE_PATH = os.path.join(DATA_PATH, '_cache')
    if not os.path.exists(CACHE_PATH):
        os.mkdir(CACHE_PATH)
    shutil.copyfile(os.path.join(DATASET_PATH, 'categories.py'), os.path.join(CACHE_PATH, 'categories.py'))
    normalize()