import matplotlib

from tasks.srdiff import SRDiffTrainer
from utils.dataset import SRDataSet
import skimage.color as sc
matplotlib.use('Agg')
import imageio
from PIL import Image
from torchvision import transforms
import random
from utils.matlab_resize import imresize
from utils.hparams import hparams
import numpy as np
import os
import imageio
from skimage import transform
import torch
from glob import glob
from models import common
class IxiDataSet2(SRDataSet):
    def __init__(self, prefix='train'):
        # super().__init__('train' if prefix == 'train' else 'test')
        self.patch_size = hparams['patch_size']
        # self.patch_size_lr = hparams['patch_size'] // hparams['sr_scale']
        # self.scale = hparams['sr_scale']
        if prefix == 'valid':
            self.len = hparams['eval_batch_size'] * hparams['valid_steps']
        # 验证设置 只验证部分图片
        self.data_aug_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20, resample=Image.BICUBIC),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ])
        # 生成lr,hr列表

        
        if prefix=='train':
            self.train1=True
            self.images_hr=sorted(glob(hparams["train_hr"]))
            self.len = len(self.images_hr)
            # self.images_lr=sorted(glob(hparams["train_lr"]))
            n_patches = 32*2000000#args.batch_size * args.test_every
            n_images = len(self.images_hr)
            self.repeat = max(n_patches//n_images,1)# TODO
        else:
            self.train1=False
            self.images_hr=sorted(glob(hparams["test_hr"]))
            self.images_lr=sorted(glob(hparams["test_lr"]))
            # TODO 数量判定
            ## n_patches = 
    def __getitem__(self, index):
        if self.train1:
            hr, filename = self._load_file(index)
            
            pair = self.get_patch_double(hr)
            # h,w,_ = pair[1].shape
            
            # lr_up = transform.resize(lr, (h, w),order=3)
            # pair.append(lr_up)
            hr = set_channel(*pair,n_channels=3)
            pair_t = np2Tensor(*pair, rgb_range=255)
            return {'img_hr': pair_t[0],'img_hr_all':pair_t,
                'item_name': filename}
        else:
            lr,hr ,filename = self._load_file(index)
            # pair = self.get_patch(hr,lr)
            hr ,lr = set_channel(hr,lr,n_channels=3)
            pair_t = np2Tensor(hr,lr, rgb_range=255)
            img_hr_all = [hr,hr]
            return {'img_hr': pair_t[1],'img_lr':pair_t[0], 'img_hr_all':img_hr_all,
                'item_name': filename}
    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        # f_lr = self.images_lr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        
        hr = imageio.imread(f_hr)
        # lr = np.load(f_lr)
        if not self.train1:
            f_lr = self.images_lr[idx]
            lr = imageio.imread(f_lr)
            return lr,hr,filename
        # h,w = hr.shape

        # hr = np.expand_dims(hr,axis=2)
        # lr = np.expand_dims(lr,axis=2)
        # lr = transform.resize(hr, (h//self.scale, w//self.scale),order=3)
        return  hr, filename
    def _get_index(self, idx):
        if self.train1:
            
            return idx % len(self.images_hr)
        else:
            return idx
    def __len__(self):
        if self.train1:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def get_patch(self, hr):
        # scale = hparams['sr_scale']# TODO
        # if self.train:
        hr = get_patch(
            hr,
            patch_size=self.patch_size,
            scale=hparams['sr_scale'],
            multi=False,
            input_large=False
        )
            # if not self.args.no_augment: lr, hr = augment(lr, hr)# TODO 数据扩增
        # else:
        #     ih, iw = lr.shape[:2]
        #     hr = hr[0:ih * scale, 0:iw * scale]
    
        return [hr]
    def get_patch_double(self, hr):
        scale = hparams['scale']
        if self.train1:
            out = []
            hr = augment(hr) #if not self.args.no_augment else hr
            # extract two patches from each image
            for _ in range(2):
                hr_patch = get_patch(
                    hr,
                    patch_size=hparams['patch_size'],
                    scale=scale,input_large=True
                )
                out.append(hr_patch)
        else:
            out = [hr]
        return out
class IxiDataSet(SRDataSet):
    def __init__(self, prefix='train'):
        super().__init__('train' if prefix == 'train' else 'test')
        self.patch_size = hparams['patch_size']
        self.patch_size_lr = hparams['patch_size'] // hparams['sr_scale']
        if prefix == 'valid':
            self.len = hparams['eval_batch_size'] * hparams['valid_steps']

        self.data_aug_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20, resample=Image.BICUBIC),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ])
        self.img_nums=self.len
        if prefix == 'train':
            self.len=int(1*1e8)
    def __getitem__(self, index):
        index=index%self.img_nums
        item = self._get_item(index)
        hparams = self.hparams
        sr_scale = hparams['sr_scale']

        img_hr = item['img']
        img_lr = item['img_lr']

        # TODO: clip for SRFlow
        h, w, c = img_hr.shape
        h = h - h % (sr_scale * 2)
        w = w - w % (sr_scale * 2)
        h_l = h // sr_scale
        w_l = w // sr_scale
        img_hr = img_hr[:h, :w]
        img_lr = img_lr[:h_l, :w_l]
        # random crop
        if self.prefix == 'train':
            if self.data_augmentation and random.random() < 0.5:
                img_hr, img_lr = self.data_augment(img_hr, img_lr)
            i = random.randint(0, h - self.patch_size) // sr_scale * sr_scale
            i_lr = i // sr_scale
            j = random.randint(0, w - self.patch_size) // sr_scale * sr_scale
            j_lr = j // sr_scale
            img_hr = img_hr[i:i + self.patch_size, j:j + self.patch_size]
            img_lr = img_lr[i_lr:i_lr + self.patch_size_lr, j_lr:j_lr + self.patch_size_lr]
        img_lr_up = imresize(img_lr , hparams['sr_scale'])  # np.float [H, W, C]
        img_hr, img_lr, img_lr_up = [self.to_tensor_norm(x).float() for x in [img_hr, img_lr, img_lr_up]]
        # TODO 
        return {
            'img_hr': img_hr, 'img_lr': img_lr,
            'img_lr_up': img_lr_up, 'item_name': item['item_name'],
            'loc': np.array(item['loc']), 'loc_bdr': np.array(item['loc_bdr'])
        }
        
            

    def __len__(self):
        return self.len

    def data_augment(self, img_hr, img_lr):
        sr_scale = self.hparams['sr_scale']
        #img_hr = Image.fromarray(img_hr)
        img_hr = self.data_aug_transforms(img_hr)
        img_hr = np.asarray(img_hr)  # np.uint8 [H, W, C]
        img_lr = imresize(img_hr, 1 / sr_scale)
        return img_hr, img_lr


class SRDiffIxi(SRDiffTrainer):
    def __init__(self):
        super().__init__()
        self.dataset_cls = IxiDataSet2

def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret
def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return _augment(*args)

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a[0]) for a in args]

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a[0]) for a in args]