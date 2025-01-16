import random
import numpy as np
import torch
import torch.utils.data as data
import basicir.data.transforms as transforms
from basicir.data.data_util import (paired_paths_from_folder,
                                 paired_paths_from_lmdb,
                                 paired_paths_from_meta_info_file)
from basicir.utils import FileClient, imfrombytes, img2tensor
from pathlib import Path
from torchvision.transforms import functional as TF

class Dataset_UnpairedImage(data.Dataset):
    """Unpaired Image dataset for image restoration.

    Read LQ (Low Quality) images only.
    The pair is ensured by sorted paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            geometric_augs (bool): Use geometric augmentations.
    """

    def __init__(self, opt):
        super(Dataset_UnpairedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.geometric_augs = opt.get('geometric_augs', False)

        self.lq_folder = opt['dataroot_lq']
        if 'mean' in opt:
            self.mean = opt['mean']
        else:
            self.mean = None
        if 'std' in opt:
            self.std = opt['std']
        else:
            self.std = None

        # Get paths of LQ images
        self.paths = sorted(list(Path(self.lq_folder).rglob('*')))
        self.paths = [str(p) for p in self.paths if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']]
        
        if len(self.paths) == 0:
            raise ValueError(f'No valid images found in {self.lq_folder}')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load LQ image
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path)
        img_lq = imfrombytes(img_bytes, float32=True)

        # Augmentation for training
        if self.opt.get('use_flip') or self.opt.get('use_rot'):
            img_lq = transforms.augment(img_lq, self.opt['use_flip'],
                                    self.opt['use_rot'])

        # Geometric augmentations
        if self.geometric_augs:
            # Random crop
            if random.random() < 0.5:
                crop_size = random.randint(160, 200)
                i, j, h, w = transforms.get_random_crop_params(img_lq, (crop_size, crop_size))
                img_lq = transforms.crop(img_lq, i, j, h, w)
                img_lq = transforms.resize(img_lq, (256, 256))
            
            # Random rotation
            if random.random() < 0.5:
                angle = random.randint(-45, 45)
                img_lq = transforms.rotate(img_lq, angle)
            
            # Random scaling
            if random.random() < 0.5:
                scale = random.uniform(0.8, 1.2)
                new_h = int(img_lq.shape[0] * scale)
                new_w = int(img_lq.shape[1] * scale)
                img_lq = transforms.resize(img_lq, (new_h, new_w))
                img_lq = transforms.center_crop(img_lq, (256, 256))

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)

        # Normalize
        if self.mean is not None or self.std is not None:
            img_lq = transforms.normalize(img_lq, self.mean, self.std)

        return {'lq': img_lq, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths) 