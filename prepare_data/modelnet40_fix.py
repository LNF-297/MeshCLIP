import os
import operator
import shutil

BASE_PATH = '..'
DATA_PATH = os.path.join(BASE_PATH, 'data')
SCRIPT_PATH = os.path.join(BASE_PATH, 'prepare_data', 'modelnet40.mlx')
DATASET_PATH = os.path.join(DATA_PATH, 'ModelNet40_Processed')

labels = []
train_file = []
train_cls = []
test_file = []
test_cls = []

def convert(IN_PATH, OUT_PATH):
    os.system('meshlabserver -i ' + IN_PATH + ' -o ' + OUT_PATH + ' -s ' + SCRIPT_PATH)

def collect_labels():
    PATH = os.path.join(DATA_PATH,'ModelNet40')
    for dirs in os.listdir(PATH):
        if operator.contains(dirs,'.'):
            continue
        labels.append(dirs)

def fix_line(line):
    if line[0] == 'O' and len(line)>4:
        return line[0:3] + '\n' + line[3:]
    else:
        return line

def fix(PATH):
    if not os.path.exists(PATH+'.backup'):
        shutil.copy(PATH,PATH+'.backup')
    f = open(PATH+'.backup','r')
    line = f.readline()
    if len(line) <= 4:
        f.close()
        print('skip')
        return
    f_fixed = open(PATH, 'w')
    while line:
        f_fixed.write(fix_line(line))
        line=f.readline()
    f.close()
    f_fixed.close()
    if os.path.exists(PATH+'.backup'):
        os.remove(PATH+'.backup')
def build():
    for i in range(len(labels)):
        PATH = os.path.join(DATA_PATH,'ModelNet40',labels[i])
        TRAIN_PATH = os.path.join(PATH,'train')
        TEST_PATH = os.path.join(PATH,'test')
        for dirs in os.listdir(TRAIN_PATH):
            if not operator.contains(dirs,'.'):
                continue
            print(dirs)
            fix(os.path.join(TRAIN_PATH,dirs))

        for dirs in os.listdir(TEST_PATH):
            if not operator.contains(dirs,'.'):
                continue
            print(dirs)
            fix(os.path.join(TEST_PATH,dirs))


if __name__ == '__main__':
    if not os.path.exists(os.path.join(DATA_PATH, 'ModelNet40')):
        print('Dataset dose not exist.')
        exit()
    collect_labels()
    print(labels)
    build()