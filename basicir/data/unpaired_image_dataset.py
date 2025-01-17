import random
from pathlib import Path
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from basicir.data.data_util import unpaired_paths_from_folder
from basicir.data.transforms import random_augmentation, random_crop
from basicir.utils import FileClient, imfrombytes, img2tensor
from basicir.utils.img_util import padding

class Dataset_UnpairedImage(data.Dataset):
    """Unpaired Image dataset for image restoration.

    Read LQ (Low Quality) images only.
    The pair is ensured by 'sorted' function, so please check the name convention.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            geometric_augs (bool): Use geometric augmentations.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_UnpairedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.geometric_augs = opt.get('geometric_augs', False)

        self.lq_folder = opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        # mean and std for normalization
        if 'mean' in opt:
            self.mean = opt['mean']
        else:
            self.mean = None
        if 'std' in opt:
            self.std = opt['std']
        else:
            self.std = None

        # Generate image paths
        if isinstance(self.lq_folder, list):
            self.paths = []
            for lq_folder in self.lq_folder:
                self.paths.extend(unpaired_paths_from_folder(
                        lq_folder, 'lq',
                        self.filename_tmpl))
        else:
            self.paths = unpaired_paths_from_folder(
                folder=self.lq_folder,
                key='lq',
                filename_tmpl=self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load lq image
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path)
        img_lq = imfrombytes(img_bytes, float32=True)

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_lq = padding(img_lq, gt_size=gt_size)

            # random crop
            img_lq = random_crop(img_lq, gt_size)

            # Geometric augmentations
            if self.geometric_augs:
                img_lq = random_augmentation(img_lq)[0]
            

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'lq_path': lq_path
        }

    def __len__(self):
        return len(self.paths) 