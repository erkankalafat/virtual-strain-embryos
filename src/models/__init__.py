from .transunet import TransUNetRegressor
from .swin_unet import SwinUNetRegressor
from .pix2pixhd import Pix2PixHD
from .diffusion import ConditionalDDPM
from .style_transfer import AdaINStyleTransfer

MODEL_REGISTRY = {
    "transunet": TransUNetRegressor,
    "swin_unet": SwinUNetRegressor,
    "pix2pixhd": Pix2PixHD,
    "diffusion": ConditionalDDPM,
    "adain": AdaINStyleTransfer,
}


def build_model(config):
    model_name = config["model"]["name"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](config)
