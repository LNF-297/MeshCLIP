import os
import pandas as pd
from torch.utils.data.dataset import Dataset
from pytorch3d.io import load_obj
import random
def get_classnames(PATH):
    classnames = []
    with open(os.path.join(PATH, 'classnames.txt'), 'r') as f:
        line = f.readline().replace('\n', '').replace('\r', '')
        while line:
            classnames.append(line)
            line = f.readline().replace('\n', '').replace('\r', '')
        f.close()
    return classnames

class ModelNet40(Dataset):
    def __init__(self,annotations_file, obj_dir):
        super(ModelNet40, self).__init__()
        self.mesh_index = pd.read_csv(annotations_file)
        self.obj_dir = obj_dir
    def __len__(self):
        return len(self.mesh_index)
    def __getitem__(self, idx):
        obj_path = os.path.join(self.obj_dir,self.mesh_index.file[idx])
        label = self.mesh_index.label[idx]
        verts, faces, aus = load_obj(obj_path)
        #mesh = load_objs_as_meshes([obj_path], load_textures=False)
        mesh  = {'verts': verts, 'faces': faces.verts_idx, 'label': label}
        return mesh

class ModelNet40_fs(Dataset):
    def __init__(self,annotations_file, obj_dir, num_shot):
        super(ModelNet40_fs, self).__init__()
        self.mesh_index = pd.read_csv(annotations_file)
        self.obj_dir = obj_dir
        self.num_shot = num_shot
        self.generate()
    def generate(self):
        data_index = {}
        for i in range(len(self.mesh_index)):
            label = self.mesh_index.label[i]
            if label in data_index.keys():
                data_index[label].append(i)
            else:
                data_index[label] = [i, ]
        self.data_index_sampled = []

        for label in data_index.keys():
            self.data_index_sampled += random.sample(data_index[label], self.num_shot)

    def __len__(self):
        return len(self.data_index_sampled)
    def __getitem__(self, idx):
        idx = self.data_index_sampled[idx]
        obj_path = os.path.join(self.obj_dir,self.mesh_index.file[idx])
        label = self.mesh_index.label[idx]
        verts, faces, aus = load_obj(obj_path)
        #mesh = load_objs_as_meshes([obj_path], load_textures=False)
        mesh  = {'verts': verts, 'faces': faces.verts_idx, 'label': label}
        return mesh