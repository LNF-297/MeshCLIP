import argparse
import os.path
from trainer.zeroshot import zeroshot
from trainer.fewshot import fewshot
from clip import clip
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parse = argparse.ArgumentParser('Model')
    parse.add_argument('--exp_name', type=str, default='exp', help='The name of the experiment')
    parse.add_argument('--trainer', type=str, default='fewshot', help='Trainer [default: fewshot]')
    parse.add_argument('--dataset_name', type=str, default='MoldelNet40', help='choose the mesh dataset. [default: MoldelNet40]')
    parse.add_argument('--device', type=str, default='cuda', help='Device [default: cuda]')
    parse.add_argument('--clip_model', type=str, default='ViT-L/14@336px', help='Available clip models:'+str(clip.available_models())+'[default: ViT-B/32]')
    parse.add_argument('--dataset_path', type=str, default=os.path.join('.', 'data', '3D_FUTURE_img'), help='dataset path')
    parse.add_argument('--mesh_views', type=int, default=14, help='The number of views')
    parse.add_argument('--batch_size', type=int, default=8, help='batch size [default: 8]')
    parse.add_argument('--num_workers', type=int, default=8, help='num_workers [default: 8]')
    parse.add_argument('--epoch', type=int, default=250, help='epoch [default: 250]')
    parse.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate [default: 0.5]')
    parse.add_argument('--adapter_ratio', type=float, default=0.6, help='Visual adapter ratio for residual connection [default: 0.6]')
    parse.add_argument('--num_shot', type=int, default=16, help='Number of shots [default: 16]')
    parse.add_argument('--lr', type=float, default=0.001, help='Learning rate [default: 0.001]')
    parse.add_argument('--load', type=str, default=None, help='The path of state_dict to load before training or testing. [default: None]')
    parse.add_argument('--in_features', type=int, default=768, help='The size of image feature for adapter. [default: 768]')
    parse.add_argument('--drop_last', type=bool, default=False, help='Whether to drop the last batch or not. [defulat: False]')
    return parse.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.trainer == 'zeroshot':
        zeroshot(args)
    if args.trainer == 'fewshot':
        fewshot(args)