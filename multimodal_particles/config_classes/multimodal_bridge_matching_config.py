import yaml
import json
from dataclasses import dataclass, field,asdict
from typing import Optional, Dict, List, Union

@dataclass
class TrainingConfig:
    batch_size: int = 1024
    data_split_frac: List[float] = field(default_factory=lambda: [0.8, 0.2, 0.0])
    epochs: int = 200
    gradient_clip_val: float = 1.0
    optimizer_name: str = "AdamW"
    lr: float = 0.001
    weight_decay: float = 5.0e-5
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1.e-8
    amsgrad: bool = False
    scheduler_name: str = "CosineAnnealingLR"
    scheduler_params: Dict[str, Union[float, int]] = field(default_factory=lambda: {
        "T_max": 1000,
        "eta_min": 5.0e-5,
        "last_epoch": -1
    })

@dataclass
class DataConfig:
    target_name: str = "AspenOpenJets"
    target_path: List[str] = field(default_factory=lambda: None)
    target_preprocess_continuous: str = "standardize"
    target_preprocess_discrete: str = "tokens"
    target_info: Dict[str, Union[list, dict]] = field(default_factory=lambda: {
        "stats": None, 
        "hist_num_particles": None # dict with histogram of number of particles
    })
    source_name: str = "GaussNoise"
    source_path: List[str] = field(default_factory=lambda: None)
    source_preprocess_continuous: str = None
    source_preprocess_discrete: str = "tokens"
    source_info: Dict[str, Union[list, dict]] = field(default_factory=lambda: {
        "stats": None, 
        "hist_num_particles": None # dict with histogram of number of particles
    })
    source_masks_from_target_masks: bool = True # if True, source mask is sampled from multinomial dist from number of target particles
    min_num_particles: int=0
    max_num_particles: int=128
    num_jets: int=1000
    dim_features_continuous: int = 3
    dim_features_discrete: int = 1
    dim_context_continuous: int = 0
    dim_context_discrete: int = 0
    vocab_size_features: int = 8
    vocab_size_context: int = 0

@dataclass
class BridgeConfig:
    continuous: str = "LinearUniformBridge"
    discrete: str = "TelegraphBridge"
    sigma: float = 0.0001
    gamma: float = 0.125
    num_timesteps: int = 1000
    time_eps: float = 0.0001

@dataclass
class EncoderConfig:
    name: str = "MultiModalEPiC"
    num_blocks: int = 2
    embedding_time: str = "SinusoidalPositionalEncoding"
    embedding_features_continuous: str = "Linear"
    embedding_features_discrete: str = "Embedding"
    embedding_context_continuous: Optional[str] = None
    embedding_context_discrete: Optional[str] = None
    dim_hidden_local: int = 16
    dim_hidden_glob: int = 16
    dim_emb_time: int = 16
    dim_emb_features_continuous: int = 16
    dim_emb_features_discrete: int = 16
    dim_emb_context_continuous: int = 0
    dim_emb_context_discrete: int = 0
    skip_connection: bool = True
    dropout: float = 0.1
    activation: str = "SELU"
    add_discrete_head: bool = True

@dataclass
class MultimodalBridgeMatchingConfig:
    name_str: str = "ExampleModel"
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)

    @staticmethod
    def from_yaml(file_path: str) -> "MultimodalBridgeMatchingConfig":
        """Initializes the class from a YAML file."""
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)
        return MultimodalBridgeMatchingConfig(
            name_str=config_dict.get("name_str", "ExampleModel"),
            bridge=BridgeConfig(**config_dict["bridge"]),
            data=DataConfig(**config_dict["data"]),
            encoder=EncoderConfig(**config_dict["encoder"]),
            train=TrainingConfig(**config_dict["train"])
        )

    def to_yaml(self, file_path: str):
        """Saves the class to a YAML file."""
        with open(file_path, "w") as file:
            yaml.dump(asdict(self), file, default_flow_style=False)