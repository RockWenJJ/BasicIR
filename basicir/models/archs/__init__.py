import importlib
from os import path as osp

from basicir.utils import scandir

from .waterformer_arch import WaterFormer
from .crossspaceformer_arch import CrossSpaceFormer
from .restormer_arch import Restormer
from .unet_arch import UNet
from .yolov8unet_arch import YOLOv8UNet
from .liteenhancenet_arch import LiteEnhanceNet
from .lu2net_arch import LU2Net, LU2Net_Bottleneck
from .luuie_arch import LUUIEv1, LUUIEv2, LUUIEv3, LUUIEv4, LUUIEv5, LUUIEv6, LUUIEv7, LUUIEv8
from .shallowuwnet_arch import ShallowUWNet
from .ssduie_arch import SSDUIE, SSDNoWave, SSDNoHist
from .waveresneth_arch import WaveResNetH
from .lu2nethybrid_arch import LU2NetHybrid


# automatically scan and import arch modules
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder)
    if v.endswith('_arch.py')
]
# import all the arch modules
_arch_modules = [
    importlib.import_module(f'basicir.models.archs.{file_name}')
    for file_name in arch_filenames
]


def dynamic_instantiation(modules, cls_type, opt):
    """Dynamically instantiate class.

    Args:
        modules (list[importlib modules]): List of modules from importlib
            files.
        cls_type (str): Class type.
        opt (dict): Class initialization kwargs.

    Returns:
        class: Instantiated class.
    """

    for module in modules:
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f'{cls_type} is not found.')
    return cls_(**opt)


def define_network(opt):
    network_type = opt.pop('type')
    net = dynamic_instantiation(_arch_modules, network_type, opt)
    return net

__all__ = [
    'UNet',
    'LUUIEv1',
    'LUUIEv2',
    'LUUIEv3',
    'LUUIEv4',
    'LUUIEv5',
    'LUUIEv6',
    'LUUIEv7',
    'LUUIEv8',
    'ShallowUWNet',
    'WaterFormer',
    'CrossSpaceFormer',
    'Restormer',
    'YOLOv8UNet',
    'LiteEnhanceNet',
    'LU2Net',
    'LU2Net_Bottleneck',
    'WaveResNetH',
    'LU2NetHybrid',
    # ... other architectures ...
]
