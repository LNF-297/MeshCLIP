import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from clip import clip
from dataset.Manifold40 import Manifold40, Manifold40_fs
from dataset.Manifold40 import get_classnames as get_calssnames_Manifold40
from dataset.ModelNet40 import ModelNet40, ModelNet40_fs
from dataset.ModelNet40 import get_classnames as get_calssnames_ModelNet40
from dataset._3D_FUTURE_sub_categories import dataset_fs as _3D_FUTURE_sub
from dataset._3D_FUTURE_sub_categories import get_classnames_fs as get_classname_3D_FUTURE_sub
from dataset._3D_FUTURE_super_categories import dataset_fs as _3D_FUTURE_super
from dataset._3D_FUTURE_super_categories import get_classnames as get_classname_3D_FUTURE_super
from utils.utils_projection_fs1 import projection as projection1
from utils.utils_projection_fs2 import projection as projection2
from pytorch3d.datasets.utils import collate_batched_meshes
from tqdm import tqdm
import numpy as np
import torch.optim as optim

TEMPLATES = {
    'Manifold40': 'a mesh photo of a {}.',
    'ModelNet40': 'a mesh photo of a {}.',
    '3D-FUTURE_sub': 'a picture of a {}.',
    '3D-FUTURE_super': 'a picture of a {}.'
}

WEIGHT = {
    'Manifold40': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float),
    'ModelNet40': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float),
    '3D-FUTURE_sub': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float),
    '3D-FUTURE_super': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
}

def load_clip(args):
    return clip.load(name=args.clip_model, device=args.device, jit=True)


def smooth_loss(pred, gold):
    eps = 0.2
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    loss = -(one_hot * log_prb).sum(dim=1).mean()
    return loss


class BatchNormPoint(nn.Module):
    def __init__(self, feat_size, sync_bn=False):
        super().__init__()
        self.feat_size = feat_size
        self.sync_bn = sync_bn
        if self.sync_bn:
            self.bn = BatchNorm2dSync(feat_size)
        else:
            self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        if self.sync_bn:
            # 4d input for BatchNorm2dSync
            x = x.view(s1 * s2, self.feat_size, 1, 1)
            x = self.bn(x)
        else:
            x = x.view(s1 * s2, self.feat_size)
            x = self.bn(x)
        return x.view(s1, s2, s3)

class Textual_Encoder(nn.Module):

    def __init__(self, args, classnames, clip_model):
        super().__init__()
        self.args = args
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = 'half'

    def forward(self):
        temp = TEMPLATES[self.args.dataset_name]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.cuda()
        text_feat = self.clip_model.encode_text(prompts)
        return text_feat

class merge(nn.Module):
    def __init__(self, args):
        super(merge, self).__init__()
        self.weight = nn.Parameter(torch.from_numpy(WEIGHT[args.dataset_name]/np.sum(WEIGHT[args.dataset_name])).to(args.device), requires_grad=True)
        self.weight.requires_grad_(True)
    def forward(self, x):
        merged_x = torch.matmul(self.weight, x)
        return merged_x

class MeshCLIP_Model(nn.Module):

    def __init__(self, args, classnames, clip_model):
        super().__init__()
        self.args = args
        # Encoders from CLIP
        self.visual_encoder = clip_model.visual
        self.textual_encoder = Textual_Encoder(args, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale

        # Multi-view projection
        self.num_views = args.mesh_views
        if args.dataset_name == 'ModelNet40' or args.dataset_name == 'Manifold40':
            self.projection = projection1(args)
        elif args.dataset_name == '3D-FUTURE_sub' or args.dataset_name == '3D-FUTURE_super':
            self.projection = projection2(args)

        # Adapter
        self.adapter = Adapter(args).half()

        self.merge = merge(args).half()
    def forward(self, mesh, batch_size):
        # Project to multi-view depth maps
        images = self.projection(mesh, batch_size)[:, :, :, 0:3]
        images = torch.permute(images, (0, 3, 1, 2)).half()

        # Image features

        image_feat = self.visual_encoder(images)
        image_feat = self.adapter(image_feat)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        # Text features
        text_feat = self.textual_encoder()
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        # Classification logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_feat @ text_feat.t() * 1.
        similarity = torch.reshape(logits, (batch_size, self.args.mesh_views, -1))
        merged_similarity = self.merge(similarity)
        return merged_similarity

class Attention(nn.Module):
    def __init__(self,args):
        super(Attention, self).__init__()
        self.num_views = args.mesh_views
        self.in_features = args.in_features
        # Self Attention
        self.q_conv = nn.Conv1d(args.mesh_views, args.mesh_views, 1, bias=False)
        self.k_conv = nn.Conv1d(args.mesh_views, args.mesh_views, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(args.mesh_views, args.mesh_views, 1)
        self.normp = BatchNormPoint(self.in_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feat):
        img_feat = feat.reshape(-1, self.num_views, self.in_features)
        img_feat = self.normp(img_feat)
        x_q = self.q_conv(img_feat).permute(0, 2, 1)
        x_k = self.k_conv(img_feat)
        x_v = self.v_conv(img_feat)
        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        ksa = torch.bmm(x_v, attention)
        feature_saout = img_feat + ksa
        return feature_saout

class Adapter(nn.Module):
    """
    Inter-view Adapter
    """

    def __init__(self, args):
        super().__init__()

        self.num_views = args.mesh_views
        self.in_features = args.in_features
        self.adapter_ratio = args.adapter_ratio
        self.fusion_init = 0.5
        self.dropout = args.dropout_rate

        self.fusion_ratio = nn.Parameter(torch.tensor([self.fusion_init] * self.num_views), requires_grad=True)
        self.fusion_ratio_res = nn.Parameter(torch.tensor([self.fusion_init] * self.num_views), requires_grad=True)

        self.global_f = nn.Sequential(
            BatchNormPoint(self.in_features),
            nn.Dropout(self.dropout),
            nn.Flatten(),
            nn.Linear(in_features=self.in_features * self.num_views,
                      out_features=self.in_features * self.num_views),
            nn.BatchNorm1d(self.in_features * self.num_views),
            nn.ReLU(),
            nn.Dropout(self.dropout))
        self.norm = nn.Sequential(
            BatchNormPoint(self.in_features),
            nn.Dropout(self.dropout),
            nn.Flatten(),
            nn.BatchNorm1d(self.in_features * self.num_views)
        )
        self.view_f = nn.Sequential(
            nn.BatchNorm1d(self.in_features * self.num_views),
            nn.Linear(in_features=self.in_features * self.num_views,
                      out_features=self.in_features * self.num_views),
            nn.ReLU(),
            nn.Linear(in_features=self.in_features * self.num_views,
                      out_features=self.in_features * self.num_views),
            nn.ReLU())

        self.attention = Attention(args)
    def forward(self, feat):
        img_feat = feat.reshape(-1, self.num_views, self.in_features)
        res_feat = feat.reshape(-1, self.num_views * self.in_features)

        # Attention Feature
        att_feat = self.attention(img_feat)
        # Global feature
        global_feat = self.global_f(att_feat * self.fusion_ratio.reshape(1, -1, 1))
        global_feat = global_feat * self.adapter_ratio + self.norm(att_feat * self.fusion_ratio_res.reshape(1, -1, 1)) * (1 - self.adapter_ratio)
        # View-wise adapted features
        view_feat = self.view_f(global_feat)

        img_feat = view_feat * self.adapter_ratio + res_feat * (1 - self.adapter_ratio)

        return img_feat.reshape(-1, self.in_features)

def check(rez, label):
    rez = torch.argmax(rez, 1, keepdim=False)
    cmp = rez == label
    cmp = cmp.int()
    cnt = np.sum(cmp.cpu().numpy())
    return cnt

def fewshot2(args):
    # create exp directory
    if not os.path.exists(os.path.join('.', 'exp')):
        os.mkdir(os.path.join('.', 'exp'))
    exp_dir = os.path.join('.', 'exp', args.exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    # save parameters
    with open(os.path.join(exp_dir, 'args.txt'), 'w') as f:
        f.writelines(str(args)+'\n')
        f.close()
    with open(os.path.join(exp_dir, 'parameters.txt'), 'w') as f:
        f.writelines(str(WEIGHT) + '\n')
        f.writelines(str(TEMPLATES) + '\n')
        f.close()
    shutil.copyfile(os.path.join('.', 'trainer', 'fewshot.py'), os.path.join(exp_dir, 'fewshot.py'))
    shutil.copyfile(os.path.join('.', 'utils', 'utils_projection_fs1.py'), os.path.join(exp_dir, 'utils_projection_fs1.py'))
    shutil.copyfile(os.path.join('.', 'utils', 'utils_projection_fs2.py'), os.path.join(exp_dir, 'utils_projection_fs2.py'))

    # load clip model
    clip_model, clip_preprocess = load_clip(args)
    args.resolution = clip_model.input_resolution.item()
    # load dataset
    if args.dataset_name == 'ModelNet40':
        classnames = get_calssnames_ModelNet40(args.dataset_path)
        test_dataloader = DataLoader(
            dataset=ModelNet40(os.path.join(args.dataset_path, 'test', 'test_files.csv'),
                               os.path.join(args.dataset_path, 'test')),
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_batched_meshes, drop_last=False,
            num_workers=args.num_workers)
        train_dataloader = DataLoader(
            dataset=ModelNet40_fs(os.path.join(args.dataset_path, 'train', 'train_files.csv'),
                                  os.path.join(args.dataset_path, 'train'), args.num_shot),
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_batched_meshes, drop_last=False,
            num_workers=args.num_workers)
    elif args.dataset_name == 'Manifold40':
        classnames = get_calssnames_Manifold40(args.dataset_path)
        test_dataloader = DataLoader(
            dataset=Manifold40(os.path.join(args.dataset_path, 'test', 'test_files.csv'),
                               os.path.join(args.dataset_path, 'test')),
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_batched_meshes, drop_last=False,
            num_workers=args.num_workers)
        train_dataloader = DataLoader(
            dataset=Manifold40_fs(os.path.join(args.dataset_path, 'train', 'train_files.csv'),
                                  os.path.join(args.dataset_path, 'train'), args.num_shot),
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_batched_meshes, drop_last=False,
            num_workers=args.num_workers)
    elif args.dataset_name == '3D-FUTURE_sub':
        classnames = get_classname_3D_FUTURE_sub()
        trainset, testset = _3D_FUTURE_sub(args.dataset_path, args.num_shot)
        test_dataloader = DataLoader(
            dataset=testset,
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_batched_meshes, drop_last=args.drop_last,
            num_workers=args.num_workers)
        train_dataloader = DataLoader(
            dataset=trainset,
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_batched_meshes, drop_last=args.drop_last,
            num_workers=args.num_workers)
    elif args.dataset_name == '3D-FUTURE_super':
        classnames = get_classname_3D_FUTURE_super()
        trainset, testset = _3D_FUTURE_super(args.dataset_path, args.num_shot)
        test_dataloader = DataLoader(
            dataset=testset,
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_batched_meshes, drop_last=args.drop_last,
            num_workers=args.num_workers)
        train_dataloader = DataLoader(
            dataset=trainset,
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_batched_meshes, drop_last=args.drop_last,
            num_workers=args.num_workers)

    # Instantiation of nn
    net = MeshCLIP_Model(args, classnames, clip_model)
    for name, param in net.named_parameters():
        if 'adapter' not in name:
            param.requires_grad_(False)

    #Load pretrained parameters
    if not args.load is None:
        net.adapter.load_state_dict(torch.load(args.load))
        print('visual adapter parameters loaded')

    net = net.to(args.device)
    f = open(os.path.join(exp_dir,' rec.txt'), 'w')

    # Train
    print('Train:')
    f.writelines('Train:\n')
    optimizer = optim.SGD(params=net.adapter.parameters(), lr=args.lr,momentum=0.9)
    max_acc = 0
    for epoch in range(args.epoch):
        print('epoch %d:' %(epoch))
        f.writelines('epoch %d:' %(epoch))
        running_loss, cnt, tot = 0.0, 0, 0
        for data in tqdm(train_dataloader):
            mesh = data['mesh'].to(args.device)
            label = data['label']
            optimizer.zero_grad()
            out = net(mesh, len(label))
            with torch.no_grad():
                cnt += check(out, torch.tensor(label).to(args.device))
                tot += len(label)
            loss = smooth_loss(out, torch.tensor(label, device=args.device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Train loss: %f acc: %f' %(running_loss, cnt/tot))
        f.writelines('Train loss: %f acc: %f' %(running_loss, cnt/tot)+'\n')
        if epoch % 10 == 9 or epoch == 0:
            cnt, tot = 0, 0
            print('Test')
            f.writelines('Test\n')
            for i, data in enumerate(tqdm(test_dataloader)):
                mesh = data['mesh'].to(args.device)
                label = data['label']
                with torch.no_grad():
                    out = net(mesh, len(label))
                cnt += check(out, torch.tensor(label).to(args.device))
                tot += len(label)
            acc = cnt / tot
            if acc > max_acc:
                max_acc = acc
                torch.save(net.adapter.state_dict(), os.path.join(exp_dir, 'Adapter.pkl'))
            print('Test acc: ' + str(acc) + ' best_acc: ' + str(max_acc))
            f.writelines('Test acc: ' + str(acc) + ' best_acc: ' + str(max_acc) + '\n')

    print('loading the best model parameters')
    net.adapter.load_state_dict(torch.load(os.path.join(exp_dir, 'Adapter.pkl')))
    # Test
    cnt, tot = 0, 0
    print('Test:')
    f.writelines('Test\n')
    for i, data in enumerate(tqdm(test_dataloader)):
        mesh = data['mesh'].to(args.device)
        label = data['label']
        with torch.no_grad():
            out = net(mesh, len(label))
        cnt += check(out, torch.tensor(label).to(args.device))
        tot += len(label)
    acc = cnt / tot
    print('acc: ' + str(acc))
    f.writelines('acc: ' + str(acc) + '\n')
    f.close()