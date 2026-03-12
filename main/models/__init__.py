import os
import torch
import torch.nn.functional as F

from functools import partial

# from .vHeat import vHeat
from .vHeat2 import vHeat
from .vHeat4HSI import vHeat4HSI


def build_vHeat_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    
    if model_type == "vHeat":
        model = vHeat(
            in_chans=config.MODEL.VHEAT.IN_CHANS, 
            patch_size=config.MODEL.VHEAT.PATCH_SIZE, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VHEAT.DEPTHS, 
            dims=config.MODEL.VHEAT.EMBED_DIM, 
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            mlp_ratio=config.MODEL.VHEAT.MLP_RATIO,
            post_norm=config.MODEL.VHEAT.POST_NORM,
            layer_scale=config.MODEL.VHEAT.LAYER_SCALE,
            img_size=config.DATA.IMG_SIZE,
            infer_mode=config.EVAL_MODE or config.THROUGHPUT_MODE,
        )
        if config.THROUGHPUT_MODE:
            model.infer_init()
        return model
    elif model_type == "vHeat4HSI":
        model = vHeat4HSI(
            hsi_patch_size=config.DATA.PATCH_SIZE,  # 高光谱数据集的patch size，也就是输入数据的大小
            hsi_band_size=config.DATA.BANDS,  # 高光谱数据集的波段数，也就是输入数据的通道数
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VHEAT.DEPTHS, 
            dims=config.MODEL.VHEAT.EMBED_DIM, 
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            mlp_ratio=config.MODEL.VHEAT.MLP_RATIO,
            post_norm=config.MODEL.VHEAT.POST_NORM,
            layer_scale=config.MODEL.VHEAT.LAYER_SCALE,
            infer_mode=config.EVAL_MODE or config.THROUGHPUT_MODE,
        )
        if config.THROUGHPUT_MODE:
            model.infer_init()
        return model
    
    
def build_model(config, is_pretrain=False):
    model = build_vHeat_model(config, is_pretrain)
    return model
