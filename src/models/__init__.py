from .transunet import TransUNetRegressor
from .swin_unet import SwinUNetRegressor
from .pix2pixhd import Pix2PixHD
from .diffusion import ConditionalDDPM
from .style_transfer import AdaINStyleTransfer
from .convnext_unet import ConvNeXtUNet
from .dino_transunet import DINOTransUNet
from .flow_matching import ConditionalFlowMatching

MODEL_REGISTRY = {
    "transunet": TransUNetRegressor,
    "swin_unet": SwinUNetRegressor,
    "convnext_unet": ConvNeXtUNet,
    "dino_transunet": DINOTransUNet,
    "pix2pixhd": Pix2PixHD,
    "diffusion": ConditionalDDPM,
    "flow_matching": ConditionalFlowMatching,
    "adain": AdaINStyleTransfer,
}


def build_model(config):
    model_name = config["model"]["name"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](config)
