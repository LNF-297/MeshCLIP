import os
import json
import random
from torch.utils.data import Dataset, random_split
from utils.utils_obj_io import load_obj
from typing import Dict, List, Optional, Tuple
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# get_classnames for both zero shot and few shot
def get_classnames():
    from data._cache.categories import _SUPER_CATEGORIES_3D as categories
    classname = []
    for dir in categories:
        # omit class "others"
        if dir['id'] == 8:
            continue
        classname.append(dir['category'])
    return classname

# Dataset for few shot

class dataset_index(Dataset):
    def __init__(self, PATH):
        super(dataset_index, self).__init__()
        self.PATH = PATH
        from data._cache.categories import _SUPER_CATEGORIES_3D as categories
        name_id_index = {}
        for dir in categories:
            if dir['id'] == 8:  # exclude "others"
                continue
            name_id_index[dir['category']] = dir['id'] - 1
        with open(os.path.join(PATH, 'model_info.json'), 'r') as f:
            js_index = json.load(f)
            f.close()
        self.index = []
        for cat in js_index:
            classname = cat['super-category']
            if not classname in name_id_index.keys():
                continue
            if classname == 'Others':
                continue
            self.index.append({'label': name_id_index[classname], 'file': cat['model_id']})
    def __len__(self):
        return len(self.index)


    def __getitem__(self, item):
        data_info = self.index[item]
        return data_info['label'], os.path.join(self.PATH, data_info['file'], 'normalized_model.obj')

class fewshot_set(Dataset):
    def __init__(self, dataset, num_shot):
        super(fewshot_set, self).__init__()
        self.full_dataset = dataset
        self.num_shot = num_shot
        self.texture_resolution = 4
        self.index = []
        index = list(range(len(self.full_dataset)))
        random.shuffle(index)
        cnt = {}
        for i in index:
            label, file = self.full_dataset[i]
            if not label in cnt.keys():
                cnt[label] = 1
            else:
                cnt[label] += 1
            if cnt[label] > self.num_shot:
                continue
            self.index.append({'label': label, 'file': file})
    def __len__(self):
        return len(self.index)

    def _load_mesh(self, model_path) -> Tuple:
        verts, faces, aux = load_obj(
            model_path,
            create_texture_atlas=True,
            load_textures=True,
            texture_atlas_size=self.texture_resolution,
        )
        textures = aux.texture_atlas
        if textures is None:
            textures = verts.new_ones(
                faces.verts_idx.shape[0],
                self.texture_resolution,
                self.texture_resolution,
                3,
            )
        return verts, faces.verts_idx, textures

    def __getitem__(self, item):
        data_info = self.index[item]
        obj_path = data_info['file']
        verts, faces, textures = self._load_mesh(obj_path)
        return {'verts': verts, 'faces': faces, 'textures': textures, 'label': data_info['label']}


class full_set(Dataset):
    def __init__(self, dataset):
        super(full_set, self).__init__()
        self.full_dataset = dataset
        self.texture_resolution = 4
        self.index = []
        for i in range(len(self.full_dataset)):
            label, file = self.full_dataset[i]
            self.index.append({'label': label, 'file': file})
    def __len__(self):
        return len(self.index)

    def _load_mesh(self, model_path) -> Tuple:
        verts, faces, aux = load_obj(
            model_path,
            create_texture_atlas=True,
            load_textures=True,
            texture_atlas_size=self.texture_resolution,
        )
        textures = aux.texture_atlas
        if textures is None:
            textures = verts.new_ones(
                faces.verts_idx.shape[0],
                self.texture_resolution,
                self.texture_resolution,
                3,
            )
        return verts, faces.verts_idx, textures

    def __getitem__(self, item):
        data_info = self.index[item]
        obj_path = data_info['file']
        verts, faces, textures = self._load_mesh(obj_path)
        return {'verts': verts, 'faces': faces, 'textures': textures, 'label': data_info['label']}


def dataset_fs(PATH, num_shot):
    fullset = dataset_index(PATH)
    size = len(fullset)
    train_size = int(size*2/3)
    test_size = size - train_size
    trainset, testset = random_split(fullset, [train_size, test_size])
    trainset = fewshot_set(trainset, num_shot)
    testset = full_set(testset)
    return trainset, testset

# Dataset for zero shot
class dataset_zs(Dataset):
    def __init__(self, PATH):
        super(dataset_zs, self).__init__()
        self.PATH = PATH
        self.texture_resolution = 4
        from data._cache.categories import _SUPER_CATEGORIES_3D as categories
        name_id_index = {}
        for dir in categories:
            if dir['id'] == 8: # exclude "others"
                continue
            name_id_index[dir['category']] = dir['id'] - 1
        with open(os.path.join(PATH, 'model_info.json'), 'r') as f:
            js_index = json.load(f)
            f.close()
        self.index = []
        for cat in js_index:
            classname = cat['super-category']
            if classname == 'Others':
                continue
            self.index.append({'label': name_id_index[classname], 'file': cat['model_id']})
    def __len__(self):
        return len(self.index)

    def _load_mesh(self, model_path) -> Tuple:
        verts, faces, aux = load_obj(
            model_path,
            create_texture_atlas=True,
            load_textures=True,
            texture_atlas_size=self.texture_resolution,
        )
        textures = aux.texture_atlas
        if textures is None:
            textures = verts.new_ones(
                faces.verts_idx.shape[0],
                self.texture_resolution,
                self.texture_resolution,
                3,
            )


        return verts, faces.verts_idx, textures

    def __getitem__(self, item):
        data_info = self.index[item]
        obj_path = os.path.join(self.PATH, data_info['file'], 'normalized_model.obj')
        verts, faces, textures = self._load_mesh(obj_path)
        return {'verts': verts, 'faces': faces, 'textures': textures, 'label': data_info['label']}