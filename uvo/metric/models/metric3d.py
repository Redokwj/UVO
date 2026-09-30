"""
Metric3D v2 depth model wrapper with zero-torchvision stubs.
Maps camera intrinsics to a canonical pinhole (f=1000) for camera-invariant metric depth.
"""

import os
import sys
from typing import Tuple, Union, Dict, Optional
import numpy as np
from ..base import BaseDepthModel

M3D_INPUT_DEFAULT = (504, 868)
M3D_INPUT_FAST = (252, 434)  # 130 ms on Jetson Orin Nano
M3D_CANONICAL_FOCAL = 1000.0


def _install_m3d_stubs():
    """
    Lightweight stubs for mmcv and timm so Metric3D runs on Jetson/edge
    without compiled mmcv or torchvision.
    """
    import importlib.machinery as mach
    import importlib.util
    import types
    import torch.nn as nn

    def _reg(name, mod, pkg=True):
        mod.__spec__ = mach.ModuleSpec(name, None)
        if pkg:
            mod.__path__ = []
        sys.modules[name] = mod

    if importlib.util.find_spec('mmcv') is None:
        try:
            import mmengine
            mmcv = types.ModuleType('mmcv')
            utils = types.ModuleType('mmcv.utils')
            utils.collect_env = lambda: {}
            utils.get_git_hash = lambda *a, **k: ''
            utils.Config = mmengine.Config
            utils.DictAction = getattr(mmengine, 'DictAction', object)
            mmcv.utils = utils
            _reg('mmcv', mmcv)
            _reg('mmcv.utils', utils, pkg=False)
        except ImportError:
            pass

    if importlib.util.find_spec('timm') is None:
        class DropPath(nn.Module):
            def __init__(self, drop_prob=0.0):
                super().__init__()
                self.drop_prob = float(drop_prob)

            def forward(self, x):
                if self.drop_prob == 0.0 or not self.training:
                    return x
                keep = 1.0 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                return x * x.new_empty(shape).bernoulli_(keep) / keep

        timm = types.ModuleType('timm')
        models = types.ModuleType('timm.models')
        layers = types.ModuleType('timm.models.layers')
        registry = types.ModuleType('timm.models.registry')
        layers.trunc_normal_ = nn.init.trunc_normal_
        layers.DropPath = DropPath
        registry.register_model = lambda f: f
        models.layers, models.registry = layers, registry
        timm.models = models
        timm.__version__ = 'stub'
        _reg('timm', timm)
        _reg('timm.models', models)
        _reg('timm.models.layers', layers, pkg=False)
        _reg('timm.models.registry', registry, pkg=False)


class Metric3DV2(BaseDepthModel):
    """
    Metric3D v2 metric depth estimation wrapper.
    Unlike monocular relative models, Metric3D takes camera intrinsics (focal length)
    and maps the scene to a canonical pinhole model.
    """

    def __init__(
        self,
        repo_or_hub: str = 'yvanyin/metric3d',
        variant: str = 'metric3d_vit_small',
        input_size: Tuple[int, int] = M3D_INPUT_DEFAULT,
        device: str = 'cuda',
    ):
        self.repo_or_hub = repo_or_hub
        self.variant = variant
        self.input_size = tuple(input_size)
        self.device = device
        self.max_depth = 300.0

        _install_m3d_stubs()
        import importlib.util
        import torch

        self._torch = torch
        self.model = None

        if os.path.isdir(repo_or_hub) and os.path.isfile(os.path.join(repo_or_hub, 'hubconf.py')):
            if repo_or_hub not in sys.path:
                sys.path.insert(0, repo_or_hub)
            spec = importlib.util.spec_from_file_location('m3d_hubconf', os.path.join(repo_or_hub, 'hubconf.py'))
            hub = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hub)
            if not hasattr(hub, variant):
                raise RuntimeError(f"Variant {variant} not found in {repo_or_hub}/hubconf.py")
            self.model = getattr(hub, variant)(pretrain=True).to(device).eval()
        else:
            self.model = torch.hub.load(repo_or_hub, variant, pretrain=True, trust_repo=True).to(device).eval()

    def predict_depth(
        self,
        image_rgb: np.ndarray,
        intrinsics: Union[Tuple[float, float, float, float], np.ndarray, Dict[str, float]],
    ) -> np.ndarray:
        """
        Predict metric depth from RGB image and camera intrinsics.
        """
        import cv2
        torch = self._torch

        if isinstance(intrinsics, dict):
            fx = float(intrinsics['fx'])
        else:
            fx = float(intrinsics[0])

        h, w = image_rgb.shape[:2]
        ih, iw = self.input_size
        sc = min(ih / h, iw / w)
        r = cv2.resize(image_rgb, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_LINEAR)
        pad_val = [123.675, 116.28, 103.53]
        h2, w2 = r.shape[:2]
        ph, pw = ih - h2, iw - w2
        pt, pl = ph // 2, pw // 2
        r = cv2.copyMakeBorder(r, pt, ph - pt, pl, pw - pl, cv2.BORDER_CONSTANT, value=pad_val)

        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        t = torch.from_numpy(r.transpose(2, 0, 1)).float()
        t = torch.div((t - mean), std)[None].to(self.device)

        with torch.no_grad():
            d, _conf, _out = self.model.inference({'input': t})

        d = d.squeeze()
        d = d[pt:d.shape[0] - (ph - pt), pl:d.shape[1] - (pw - pl)]
        d = torch.nn.functional.interpolate(d[None, None], (h, w), mode='bilinear').squeeze()
        
        # De-canonicalization: scale by focal length of resized image
        d = d * (fx * sc / M3D_CANONICAL_FOCAL)
        depth_m = torch.clamp(d, 0, self.max_depth).float().cpu().numpy()
        return np.asarray(depth_m, dtype=np.float32)
