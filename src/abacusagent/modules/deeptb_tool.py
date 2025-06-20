import os
import json
from typing import Optional, Literal, Dict, Any, TypedDict
from pathlib import Path
from abacusagent.init_mcp import mcp

class ConfigResult(TypedDict):
    config_path: str

@mcp.tool()
def generate_deeptb_config(
    material: Literal["Si"] = "Si",
) -> ConfigResult:
    """
    生成用于 DeePTB 模型训练的配置文件。

    参数:
        config_output_path: 输出的配置 JSON 文件路径。
        material: 材料类型，目前仅支持 "Si"。

    返回:
        包含配置文件路径的字典。

    抛出:
        RuntimeError: 写入配置文件失败。
    """
    
    config_output_path = "deeptb_config.json"

    config = {
        "common_options": {
            "seed": 12342,
            "basis": {
                "Si": "2s2p1d"
            },
            "device": "cuda",
            "dtype": "float32",
            "overlap": True
        },
        "model_options": {
            "embedding": {
                "method": "slem",
                "r_max": {
                    "Si": 7.1
                },
                "irreps_hidden": "32x0e+32x1o+16x2e+8x3o+8x4e+4x5o",
                "n_layers": 3,
                "n_radial_basis": 18,
                "env_embed_multiplicity": 10,
                "avg_num_neighbors": 51,
                "latent_dim": 64,
                "latent_channels": [32],
                "tp_radial_emb": True,
                "tp_radial_channels": [32],
                "PolynomialCutoff_p": 6,
                "cutoff_type": "polynomial",
                "res_update": True,
                "res_update_ratios": 0.5,
                "res_update_ratios_learnable": False
            },
            "prediction": {
                "method": "e3tb",
                "scales_trainable": False,
                "shifts_trainable": False,
                "neurons": [64, 64]
            }
        },
        "train_options": {
            "num_epoch": 1000,
            "batch_size": 2,
            "optimizer": {
                "lr": 0.005,
                "type": "Adam"
            },
            "lr_scheduler": {
                "type": "rop",
                "factor": 0.8,
                "patience": 50,
                "min_lr": 1e-6
            },
            "loss_options": {
                "train": {"method": "hamil_abs"}
            },
            "save_freq": 1000,
            "validation_freq": 10,
            "display_freq": 1,
            "use_tensorboard": True
        },
        "data_options": {
            "train": {
                "type": "DefaultDataset",
                "root": "you data path",
                "prefix": "Si64",
                "get_Hamiltonian": True,
                "get_eigenvalues": False,
                "get_overlap": True,
                "get_DM": False
            }
        }
    }

    try:
        with open(config_output_path, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        raise RuntimeError(f"写入配置文件失败: {e}")

    return {"config_path": str(Path(config_output_path).absolute())}
