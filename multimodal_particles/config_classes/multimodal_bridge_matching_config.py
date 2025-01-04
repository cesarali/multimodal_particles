import json
import yaml
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
class EncoderConfig:
    name: str = "MultiModalEPiC"
    num_blocks: int = 2
    emb_time: int = 16
    emb_features_continuous: int = 16
    emb_features_discrete: int = 16
    emb_context_continuous: int = 0
    emb_context_discrete: int = 0
    hidden_local: int = 16
    hidden_glob: int = 16
    time_embedding: str = "SinusoidalPositionalEncoding"
    features_continuous_embedding: str = "Linear"
    features_discrete_embedding: str = "Embedding"
    context_continuous_embedding: Optional[str] = None
    context_discrete_embedding: Optional[str] = None
    skip_connection: bool = True
    dropout: float = 0.1
    activation: str = "SELU"
    add_discrete_head: bool = True

@dataclass
class MultimodalBridgeMatchingConfig:
    name_str: str = "ExampleModel"
    bridge_continuous: str = "LinearUniformBridge"
    bridge_discrete: str = "TelegraphBridge"
    bridge_params: Dict[str, float] = field(default_factory=lambda: {"sigma": 0.0001, "gamma": 0.125})
    dim_features_continuous: int = 3
    dim_features_discrete: int = 1
    dim_context_continuous: int = 0
    dim_context_discrete: int = 0
    vocab_size_features: int = 8
    vocab_size_context: int = 0

    
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    pipeline: Dict[str, float] = field(default_factory=lambda: {
        "method": "EulerLeapingSolver",
        "num_timesteps": 1000,
        "time_eps": 0.0001
    })
    train: TrainingConfig = field(default_factory=TrainingConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=4)

    @classmethod
    def from_full_config(cls,full_config):
        config = full_config.model.__dict__
        config["encoder"] = EncoderConfig(**config["encoder"].__dict__)
        config["pipeline"] = config["pipeline"].__dict__
        config["bridge_params"] = config["bridge_params"].__dict__
        config["train"].scheduler_params = config["train"].scheduler_params.__dict__
        config["train"] = TrainingConfig(**config["train"].__dict__)
        return cls(**config)
            
    @classmethod
    def from_json(cls, data: dict) -> "MultimodalBridgeMatchingConfig":
        data["encoder"] = EncoderConfig(**data["encoder"])
        return cls(**data)

    @classmethod
    def from_experiment_yaml(cls,config_source):
        with open(config_source, "r") as f:
            experiment_dict = yaml.safe_load(f)
            config = experiment_dict["model"]
            return cls.from_json(config)