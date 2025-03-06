import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
from pdb import set_trace as stx
import numpy as np

# from basicir.train import parse_options
from basicir.utils.options import parse

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def normalize_image(x):
    x_min = x.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    x_max = x.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    return (x - x_min) / (x_max - x_min)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Imge Enhancement on your own images.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint')
    parser.add_argument('--input-dir', default='./input', type=str, help='Input directory')
    parser.add_argument('--output-dir', default='./output', type=str, help='Output directory')
    parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')

    args = parser.parse_args()

    opt = parse(args.config, is_train=False)
    checkpoint = args.checkpoint
    input_dir = args.input_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']

    if any([input_dir.endswith(ext) for ext in extensions]):
        files = [input_dir]
    else:
        files = []
        for ext in extensions:
            files.extend(glob(os.path.join(input_dir, '*.'+ext)))
        files = natsorted(files)
    
    if len(files) == 0:
        raise Exception(f'No files found at {input_dir}')

    parameters = opt['network_g']

    arch_type = parameters.pop('type')
    if arch_type[-2].lower()== 'v': # for version information such as UNetv1
        arch_file = arch_type[:-2]
    else:
        arch_file = arch_type
    load_arch = run_path(f'./basicir/models/archs/{arch_file.lower()}_arch.py')
    model = load_arch[arch_type](**parameters)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    weights = torch.load(checkpoint)
    loaded_weights = weights['params_ema'] if "params_ema" in weights else weights['params']
    model.load_state_dict(loaded_weights, strict=False)
    model.eval()
    model.output_all_components = True

    img_multiple_of = 8

    with torch.no_grad():
        for file_ in tqdm(files):
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
            
            img = load_img(file_)

            input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)

            # Pad the input if not_multiple_of 8
            height,width = input_.shape[2], input_.shape[3]
            H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
            padh = H-height if height%img_multiple_of!=0 else 0
            padw = W-width if width%img_multiple_of!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
            
            try:
                if args.tile is None:
                    ## Testing on the original resolution image
                    restored, backscatter, transmission, wb = model(input_)
                else:
                    #TODO: modify the codes to make it output all components
                    # test the image tile by tile
                    b, c, h, w = input_.shape
                    tile = min(args.tile, h, w)
                    assert tile % 8 == 0, "tile size should be multiple of 8"
                    tile_overlap = args.tile_overlap

                    stride = tile - tile_overlap
                    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
                    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
                    E = torch.zeros(b, c, h, w).type_as(input_)
                    W = torch.zeros_like(E)

                    for h_idx in h_idx_list:
                        for w_idx in w_idx_list:
                            in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                            out_patch = model(in_patch)
                            out_patch_mask = torch.ones_like(out_patch)

                            E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                            W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
                    restored = E.div_(W)

                reconstructed = restored * transmission + backscatter * (1 - transmission)

                # restored = normalize_image(restored)
                # # restored_wb = normalize_image(restored * wb[..., None, None])
                # transmission = normalize_image(transmission)
                # backscatter = normalize_image(backscatter)
                # reconstructed = normalize_image(reconstructed)
                restored = torch.clamp(restored, 0, 1)
                transmission = torch.clamp(transmission, 0, 1)
                backscatter = torch.clamp(backscatter, 0, 1) * torch.ones_like(restored)
                reconstructed = torch.clamp(reconstructed, 0, 1)
                

                # Unpad the output
                restored = restored[:,:,:height,:width]
                # restored_wb = restored_wb[:,:,:height,:width]
                transmission = transmission[:,:,:height,:width]
                backscatter = backscatter[:,:,:height,:width]
                reconstructed = reconstructed[:,:,:height,:width]

                restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                # restored_wb = restored_wb.permute(0, 2, 3, 1).cpu().detach().numpy()
                transmission = transmission.permute(0, 2, 3, 1).cpu().detach().numpy()
                backscatter = backscatter.permute(0, 2, 3, 1).cpu().detach().numpy()
                reconstructed = reconstructed.permute(0, 2, 3, 1).cpu().detach().numpy()

                restored = img_as_ubyte(restored[0])
                # restored_wb = img_as_ubyte(restored_wb[0])
                transmission = img_as_ubyte(transmission[0])
                backscatter = img_as_ubyte(backscatter[0])
                reconstructed = img_as_ubyte(reconstructed[0])

                f = os.path.splitext(os.path.split(file_)[-1])[0]

                f = os.path.split(file_)[-1]
                f_back = f + '_back.jpg'
                f_trans = f + '_trans.jpg'
                f_recon = f + '_recon.jpg'
                # f_restored_wb = f + '_restored_wb.jpg'
                save_img(os.path.join(output_dir, f), restored)
                save_img(os.path.join(output_dir, f_trans), transmission)
                save_img(os.path.join(output_dir, f_back), backscatter)
                save_img(os.path.join(output_dir, f_recon), reconstructed)
                # save_img(os.path.join(output_dir, f_restored_wb), restored_wb)
            except Exception as e:
                print(e)

    print(f'Output images saved at {output_dir}')