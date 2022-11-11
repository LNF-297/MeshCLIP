import os
import operator
import pandas as pd

BASE_PATH = '..'
DATA_PATH = os.path.join(BASE_PATH, 'data')
SCRIPT_PATH = os.path.join(BASE_PATH, 'prepare_data', 'shrec_16.mlx')
DATASET_PATH = os.path.join(DATA_PATH, 'shrec_16_Processed')

labels = []
train_file = []
train_cls = []
test_file = []
test_cls = []

def convert(IN_PATH, OUT_PATH):
    os.system('meshlabserver -i ' + IN_PATH + ' -o ' + OUT_PATH + ' -s ' + SCRIPT_PATH)

def collect_labels():
    PATH = os.path.join(DATA_PATH, 'shrec_16_merged')
    for dirs in os.listdir(PATH):
        if operator.contains(dirs, '.'):
            continue
        labels.append(dirs)
    with open(os.path.join(DATASET_PATH, 'classnames.txt'), 'w') as f:
        for name in labels:
            f.writelines(name+'\n')
        f.close()

def build():
    if not os.path.exists(os.path.join(DATASET_PATH,'train')):
        os.mkdir(os.path.join(DATASET_PATH,'train'))
    if not os.path.exists(os.path.join(DATASET_PATH,'test')):
        os.mkdir(os.path.join(DATASET_PATH,'test'))
    for i in range(len(labels)):
        PATH = os.path.join(DATA_PATH,'shrec_16_merged',labels[i])
        TRAIN_PATH = os.path.join(PATH,'train')
        TEST_PATH = os.path.join(PATH,'test')
        for dirs in os.listdir(TRAIN_PATH):
            if not operator.contains(dirs,'.'):
                continue
            print(dirs)
            convert(os.path.join(TRAIN_PATH,dirs),os.path.join(DATASET_PATH,'train',dirs))
            train_file.append(dirs)
            train_cls.append(i)
        for dirs in os.listdir(TEST_PATH):
            if not operator.contains(dirs,'.'):
                continue
            print(dirs)
            convert(os.path.join(TEST_PATH,dirs),os.path.join(DATASET_PATH,'test',dirs))
            test_file.append(dirs)
            test_cls.append(i)
    train_frame = pd.DataFrame({'file': train_file, 'label': train_cls})
    test_frame = pd.DataFrame({'file': test_file, 'label': test_cls})
    train_frame.to_csv(os.path.join(DATASET_PATH, 'train', 'train_files.csv'))
    test_frame.to_csv(os.path.join(DATASET_PATH, 'test', 'test_files.csv'))

if __name__ == '__main__':
    if not os.path.exists(os.path.join(DATA_PATH, 'shrec_16_merged')):
        print('Dataset dose not exist.')
        exit()
    if os.path.exists(DATASET_PATH):
        print('Prepared dateset already existed.')
        exit()
    os.mkdir(DATASET_PATH)
    collect_labels()
    print(labels)
    build()