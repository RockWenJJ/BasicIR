import numpy as np
import os
import random
import time
import torch
from os import path as osp
import cv2
from skimage import img_as_ubyte
import torch.nn.functional as F

from .dist_util import master_only
from .logger import get_root_logger


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


@master_only
def make_exp_dirs(opt):
    """Make dirs for experiments."""
    path_opt = opt['path'].copy()
    if opt['is_train']:
        mkdir_and_rename(path_opt.pop('experiments_root'))
    else:
        mkdir_and_rename(path_opt.pop('results_root'))
    for key, path in path_opt.items():
        if ('strict_load' not in key) and ('pretrain_network' not in key) and ('resume' not in key) and ('weights' not in key):
            os.makedirs(path, exist_ok=True)


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def scandir_SIDD(dir_path, keywords=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        keywords (str | tuple(str), optional): File keywords that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (keywords is not None) and not isinstance(keywords, (str, tuple)):
        raise TypeError('"keywords" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, keywords, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if keywords is None:
                    yield return_path
                elif return_path.find(keywords) > 0:
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, keywords=keywords, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, keywords=keywords, recursive=recursive)

def check_resume(opt, resume_iter):
    """Check resume states and pretrain_network paths.

    Args:
        opt (dict): Options.
        resume_iter (int): Resume iteration.
    """
    logger = get_root_logger()
    if opt['path']['resume_state']:
        # get all the networks
        networks = [key for key in opt.keys() if key.startswith('network_')]
        flag_pretrain = False
        for network in networks:
            if opt['path'].get(f'pretrain_{network}') is not None:
                flag_pretrain = True
        if flag_pretrain:
            logger.warning(
                'pretrain_network path will be ignored during resuming.')
        # set pretrained model paths
        for network in networks:
            name = f'pretrain_{network}'
            basename = network.replace('network_', '')
            if opt['path'].get('ignore_resume_networks') is None or (
                    basename not in opt['path']['ignore_resume_networks']):
                opt['path'][name] = osp.join(
                    opt['path']['models'], f'net_{basename}_{resume_iter}.pth')
                logger.info(f"Set {name} to {opt['path'][name]}")


def sizeof_fmt(size, suffix='B'):
    """Get human readable file size.

    Args:
        size (int): File size.
        suffix (str): Suffix. Default: 'B'.

    Return:
        str: Formated file siz.
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return f'{size:3.1f} {unit}{suffix}'
        size /= 1024.0
    return f'{size:3.1f} Y{suffix}'

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def restore_single_img(model, img_path, tile=None, tile_overlap=32, max_tile_size=384):
    """Restore a single image.
    
    Args:
        model: The restoration model
        img_path (str): Path to input image
        tile (int): Tile size for splitting large images. If None, process whole image at once
        tile_overlap (int): Overlap between tiles to avoid boundary artifacts
        max_tile_size (int): Maximum tile size to prevent OOM errors
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    img_multiple_of = 8

    # Clear GPU memory before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    img = load_img(img_path)
    input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)

    # Pad the input if not multiple of 8
    height, width = input_.shape[2], input_.shape[3]
    H, W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
    padh = H-height if height%img_multiple_of!=0 else 0
    padw = W-width if width%img_multiple_of!=0 else 0
    input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

    try:
        if tile is None and max(height, width) <= max_tile_size:
            # Process whole image if it's small enough
            with torch.no_grad():
                restored = model(input_)
        else:
            # Automatically determine tile size if not specified
            if tile is None:
                tile = min(max_tile_size, max(height, width))
            tile = min(tile, max_tile_size)
            
            # Ensure tile size is multiple of 8
            tile = (tile // img_multiple_of) * img_multiple_of
            
            # Process image tile by tile
            b, c, h, w = input_.shape
            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h, w).type_as(input_)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    # Clear cache before processing each tile
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    
                    # Process tile with gradient disabled
                    with torch.no_grad():
                        out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                    W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
                    
                    # Explicitly delete unnecessary tensors
                    del out_patch, out_patch_mask
                    
            restored = E.div_(W)

        # Normalize each channel
        for c in range(3):
            c_data = restored[:, c, :, :]
            c_min = c_data.min()
            c_max = c_data.max()
            restored[:, c, :, :] = (c_data - c_min) / (c_max - c_min + 1e-7)

        # Unpad the output
        restored = restored[:,:,:height,:width]

        # Convert to numpy array
        restored = restored.cpu()
        restored = restored.permute(0, 2, 3, 1).numpy()
        restored = img_as_ubyte(restored[0])
        
        # Clear GPU memory after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as error:
        print(f"Error during image restoration: {error}")
        return img
    
    return restored