# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse

from PIL import Image
import numpy as np

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

sys.path.append("../network")
import utils
import vision_transformer as vits

#sys.path.append("../dataset")
#from camelyon16c import RandomDistortions


class AutoPad(torch.nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        w, h = img.size[0], img.size[1]
        pad_h = self.size - h
        pad_w = self.size - w
        pad = pth_transforms.Pad((0, 0, pad_w, pad_h), fill=255)
        return pad(img)


class ImagePyramidDataset(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        target_downsamples=[[1, 'x20']],
        transform = None,
        outd = ''):
        super().__init__(root, transform=transform)

        self.transform = transform

        self.samples = []
        files = os.listdir(root)
        for f in files:
            if not os.path.isdir(os.path.join(root, f)):
                continue
            dirs = os.listdir(os.path.join(root, f))
            dirs = sorted([int(_) for _ in dirs], reverse=True)
            for ds in target_downsamples:
                in_path = os.path.join(root, f, str(dirs[ds[0]]))
                out_path = os.path.join(f, ds[1])
                for img_name in os.listdir(in_path):
                    if (ds[1] == 'x20' or ds[1] == 'x40') and os.path.getsize(os.path.join(in_path, img_name)) < 5000:
                        continue
                    if outd != '' and os.path.exists(os.path.join(outd, out_path)):
                        continue
                    self.samples.append([os.path.join(in_path, img_name), out_path, \
                        os.path.join(out_path, img_name.replace('.jpeg', '.npy'))])

    def __getitem__(self, index: int):

        in_path, out_dir, out_path = self.samples[index]
        img = Image.open(in_path).convert("RGB")
        img = self.transform(img)
        return img, out_dir, out_path


def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        AutoPad(256),
        pth_transforms.FiveCrop(224),
        #RandomDistortions(), # camelyon16-c
        pth_transforms.Lambda(lambda crops: torch.stack([pth_transforms.ToTensor()(crop) for crop in crops])),
        pth_transforms.Lambda(lambda crops: torch.stack([pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(crop) for crop in crops])),
    ])
    dataset = ImagePyramidDataset(args.data_path, [[0, 'x20']], transform, args.dump_features)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, None, args.arch, args.patch_size)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features ...")
    extract_features(model, data_loader, args.use_cuda, args.dump_features)


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, dump_features=''):
    metric_logger = utils.MetricLogger(delimiter="  ")
    for imgs, out_dirs, out_paths in metric_logger.log_every(data_loader, 10):
        imgs = imgs.cuda(non_blocking=True)
        bs, nc, c, h, w = imgs.size()
        feats = model(imgs.reshape(-1, c, h, w))
        feats = feats.reshape(bs, nc, -1).mean(1).cpu().numpy()
        
        for i in range(len(out_dirs)):
            od = os.path.join(dump_features, out_dirs[i])
            os.makedirs(od, exist_ok=True)
            np.save(os.path.join(dump_features, out_paths[i]), feats[i])
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract feature for image pyramid')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--dump_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    extract_feature_pipeline(args)

    dist.barrier()
