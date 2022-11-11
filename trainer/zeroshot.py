"""
zero shot for ModelNet40, Manifold40 and 3D-FUTURE
"""
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from clip import clip
from dataset.Manifold40 import Manifold40
from dataset.Manifold40 import get_classnames as get_calssnames_Manifold40
from dataset.ModelNet40 import ModelNet40
from dataset.ModelNet40 import get_classnames as get_calssnames_ModelNet40
from dataset._3D_FUTURE_sub_categories import dataset_zs as _3D_FUTURE_sub
from dataset._3D_FUTURE_sub_categories import get_classnames_zs as get_classname_3D_FUTURE_sub
from dataset._3D_FUTURE_super_categories import dataset_zs as _3D_FUTURE_super
from dataset._3D_FUTURE_super_categories import get_classnames as get_classname_3D_FUTURE_super
from utils.utils_projection_zs1 import projection as projection1
from utils.utils_projection_zs2 import projection as projection2
from pytorch3d.datasets.utils import collate_batched_meshes
from tqdm import tqdm
import numpy as np

TEMPLATES = {
    'Manifold40': 'a picture of a {}',
    'ModelNet40': 'a picture of a {}.',
    '3D-FUTURE_sub': 'a picture of a {}.',
    '3D-FUTURE_super': 'a picture of a {}.'

}

WEIGHT = {
    'Manifold40': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float),
    'ModelNet40': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float),
    '3D-FUTURE_sub': np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=float),
    '3D-FUTURE_super': np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
}

def load_clip(args):
    return clip.load(name=args.clip_model, device=args.device, jit=True)

class Textual_Encoder(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super(Textual_Encoder, self).__init__()
        self.args = args
        self.classnames = classnames
        self.clip_model = clip_model

    def forward(self):
        template = TEMPLATES[self.args.dataset_name]
        prompts = [template.format(classname.replace('_',' ')) for classname in self.classnames]
        prompts = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(self.args.device)
        text_features = self.clip_model.encode_text(prompts)
        return text_features

class Image_Encoder(nn.Module):
    def __init__(self, args, clip_model):
        super(Image_Encoder, self).__init__()
        self.model = clip_model
        self.args = args

    def forward(self, image):
        image_features = self.model.encode_image(image)
        return image_features

class MeshCLIP_zs(nn.Module):
    def __init__(self, args, classnames,clip_model):
        super(MeshCLIP_zs,self).__init__()
        self.args = args
        if args.dataset_name == 'ModelNet40' or args.dataset_name == 'Manifold40':
            self.projection = projection1(args)
        elif args.dataset_name == '3D-FUTURE_sub' or args.dataset_name == '3D-FUTURE_super':
            self.projection = projection2(args)
        self.image_encoder = Image_Encoder(args, clip_model)
        text_encoder = Textual_Encoder(args, classnames, clip_model)
        self.text_features = text_encoder()
        self.weight = torch.from_numpy(WEIGHT[args.dataset_name]).half().to(args.device)
    def forward(self, mesh, batch_size):
        images = self.projection(mesh, batch_size)[:, :, :, 0:3]
        images = torch.permute(images, (0, 3, 1, 2))
        image_features = self.image_encoder(images)
        text_features = self.text_features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T)
        similarity = torch.reshape(similarity, (batch_size, self.args.mesh_views, similarity.shape[1]))
        merged_similarity = torch.matmul(self.weight, similarity).softmax(dim=-1)
        return merged_similarity

def check(rez, label):
    rez = torch.argmax(rez, 1, keepdim=False)
    cmp = rez == label
    cmp = cmp.int()
    cnt = np.sum(cmp.cpu().numpy())
    return cnt

def zeroshot(args):
    # create exp directory
    if not os.path.exists(os.path.join('.', 'exp')):
        os.mkdir(os.path.join('.', 'exp'))
    exp_dir = os.path.join('.', 'exp', args.exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    # save parameters
    with open(os.path.join(exp_dir, 'args.txt'), 'w') as f:
        f.writelines(str(args) + '\n')
        f.close()
    with open(os.path.join(exp_dir, 'parameters.txt'), 'w') as f:
        f.writelines(str(WEIGHT) + '\n')
        f.writelines(str(TEMPLATES) + '\n')
        f.close()
    shutil.copyfile(os.path.join('.', 'trainer', 'zeroshot.py'), os.path.join(exp_dir, 'zeroshot.py'))
    shutil.copyfile(os.path.join('.', 'utils', 'utils_projection_zs1.py'), os.path.join(exp_dir, 'utils_projection_zs1.py'))
    shutil.copyfile(os.path.join('.', 'utils', 'utils_projection_zs2.py'), os.path.join(exp_dir, 'utils_projection_zs2.py'))
    clip_model, clip_preprocess = load_clip(args)
    args.resolution = clip_model.input_resolution.item()
    # load data
    if args.dataset_name == 'ModelNet40':
        classnames = get_calssnames_ModelNet40(args.dataset_path)
        test_dataloader = DataLoader(ModelNet40(os.path.join(args.dataset_path, 'test', 'test_files.csv'),
                                                os.path.join(args.dataset_path, 'test')), batch_size=args.batch_size,
                                     shuffle=False, collate_fn=collate_batched_meshes, drop_last=False,
                                     num_workers=args.num_workers)
    elif args.dataset_name == 'Manifold40':
        classnames = get_calssnames_Manifold40(args.dataset_path)
        test_dataloader = DataLoader(Manifold40(os.path.join(args.dataset_path, 'test', 'test_files.csv'),
                                                os.path.join(args.dataset_path, 'test')), batch_size=args.batch_size,
                                     shuffle=False, collate_fn=collate_batched_meshes, drop_last=False,
                                     num_workers=args.num_workers)
    elif args.dataset_name == '3D-FUTURE_sub':
        classnames = get_classname_3D_FUTURE_sub()
        test_dataloader = DataLoader(_3D_FUTURE_sub(args.dataset_path), batch_size=args.batch_size, shuffle=True,
                                     collate_fn=collate_batched_meshes, drop_last=False, num_workers=args.num_workers)
    elif args.dataset_name == '3D-FUTURE_super':
        classnames = get_classname_3D_FUTURE_super()
        test_dataloader = DataLoader(_3D_FUTURE_super(args.dataset_path), batch_size=args.batch_size, shuffle=True,
                                     collate_fn=collate_batched_meshes, drop_last=False, num_workers=args.num_workers)


    net = MeshCLIP_zs(args, classnames, clip_model)
    cnt = 0
    tot = 0
    for i, data in enumerate(tqdm(test_dataloader)):
        mesh = data['mesh'].to(args.device)
        label = data['label']
        with torch.no_grad():
            rez = net(mesh, len(label))
        cnt += check(rez, torch.tensor(label).to(args.device))
        tot += len(label)
    acc = cnt / tot
    print('acc: ' + str(acc))
    with open(os.path.join(exp_dir, 'accuracy.txt'), 'w') as f:
        f.writelines("acc: " + str(acc))
        f.close()