import yaml
from dataclasses import dataclass, field,asdict
from typing import List, Optional, Dict

@dataclass
class DatasetKwargs:
    class_name: str = "training.dataset.QM9Dataset"
    random_rotation: bool = False
    only_second_half: bool = True
    subset: int = -1
    pos_norm: float = 1.0
    atom_type_norm: float = 0.25
    train_or_valid: str = "train"
    condition_on_alpha: bool = True
    shuffle_node_ordering: bool = True
    charge_norm: float = 10.0

@dataclass
class DataLoaderKwargs:
    pin_memory: bool = True
    num_workers: int = 8
    prefetch_factor: int = 2

@dataclass
class LossKwargs:
    class_name: str = "training.loss.JumpLossFinalDim"
    score_loss_weight: float = 1.0
    rate_loss_weight: float = 1.0
    min_t: float = 0.001
    mean_or_sum_over_dim: str = "mean"
    nearest_atom_pred: bool = True
    rate_function_name: str = "step"
    noise_schedule_name: str = "vp_sde"
    auto_loss_weight: float = 1.0
    vp_sde_beta_max: float = 20.0
    nearest_atom_loss_weight: float = 1.0
    x0_logit_ce_loss_weight: float = 1.0
    vp_sde_beta_min: float = 0.1
    loss_type: str = "eps"
    rate_cut_t: float = 0.1

@dataclass
class OptimizerKwargs:
    class_name: str = "torch.optim.Adam"
    lr: float = 3e-5
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8

@dataclass
class StructureKwargs:
    exist: List[int] = field(default_factory=lambda: [1] * 9)
    observed: List[int] = field(default_factory=lambda: [0, 0, 0, 1, 1, 1, 1, 1, 1])

@dataclass
class SamplerKwargs:
    class_name: str = "training.sampler.JumpSampler"
    dt: float = 0.001
    do_jump_back: bool = False
    corrector_start_time: float = 0.1
    corrector_steps: int = 0
    corrector_finish_time: float = 0.003
    dt_schedule: str = "uniform"
    dt_schedule_h: float = 0.001
    condition_type: str = "sweep"
    do_jump_corrector: bool = False
    guidance_weight: float = 1.0
    dt_schedule_tc: float = 0.5
    condition_sweep_idx: int = 0
    sample_near_atom: bool = True
    do_conditioning: bool = False
    condition_sweep_path: Optional[str] = None
    dt_schedule_l: float = 0.001
    corrector_snr: float = 0.1
    jump_back_start_time: float = 0.5
    no_noise_final_step: bool = False

@dataclass
class GradConditionerKwargs:
    class_name: str = "training.grad_conditioning.MoleculeJump"
    grad_norm_clip: float = 1.0
    lr_rampup_kimg: int = 320

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
class NetworkKwargs:
    model_type: str = "EGNNMultiHeadJump"
    rate_use_x0_pred: bool = True
    transformer_dim: int = 128
    n_heads: int = 8
    n_attn_blocks: int = 8
    detach_last_layer: bool = True
    noise_embed: str = "ts*1000"
    use_fp16: bool = False
    class_name: str = "training.networks.EpsilonPrecond"
    augment_dim: int = 9

@dataclass
class AugmentKwargs:
    class_name: str = "training.augment.AugmentPipe"
    p: float = 0.12
    xflip: float = 1e8
    yflip: int = 1
    scale: int = 1
    rotate_frac: int = 1
    aniso: int = 1
    translate_frac: int = 1

@dataclass
class TransdimensionalMoleculesConfig:
    dataset_kwargs: DatasetKwargs = field(default_factory=DatasetKwargs)
    distributed: bool = False
    device: str = "cuda"
    data_loader_kwargs: DataLoaderKwargs = field(default_factory=DataLoaderKwargs)
    loss_kwargs: LossKwargs = field(default_factory=LossKwargs)
    optimizer_kwargs: OptimizerKwargs = field(default_factory=OptimizerKwargs)
    just_visualize: bool = False
    structure_kwargs: StructureKwargs = field(default_factory=StructureKwargs)
    sampler_kwargs: SamplerKwargs = field(default_factory=SamplerKwargs)
    grad_conditioner_kwargs: GradConditionerKwargs = field(default_factory=GradConditionerKwargs)
    network_kwargs: NetworkKwargs = field(default_factory=NetworkKwargs)
    augment_kwargs: AugmentKwargs = field(default_factory=AugmentKwargs)
    total_kimg: int = 200000
    ema_halflife_kimg: int = 500
    batch_size: int = 64
    batch_gpu: Optional[int] = None
    loss_scaling: float = 1.0
    cudnn_benchmark: bool = True
    kimg_per_tick: int = 50
    snapshot_ticks: int = 25
    state_dump_ticks: int = 25
    log_img_ticks: int = 50
    seed: int = 2047813205
    run_dir: str = ""

@dataclass
class TransdimensionalEpicConfig:
    dataset_kwargs: DatasetKwargs = field(default_factory=DatasetKwargs)
    data_loader_kwargs: DataLoaderKwargs = field(default_factory=DataLoaderKwargs)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    dim_features_continuous: int = 3
    dim_features_discrete: int = 1
    dim_context_continuous: int = 0
    dim_context_discrete: int = 0
    vocab_size_features: int = 8
    vocab_size_context: int = 0
    batch_size: int = 64
    min_num_particles: int =  0
    max_num_particles: int = 128
    
    loss_kwargs: LossKwargs = field(default_factory=LossKwargs)
    optimizer_kwargs: OptimizerKwargs = field(default_factory=OptimizerKwargs)
    structure_kwargs: StructureKwargs = field(default_factory=StructureKwargs)
    sampler_kwargs: SamplerKwargs = field(default_factory=SamplerKwargs)
    grad_conditioner_kwargs: GradConditionerKwargs = field(default_factory=GradConditionerKwargs)
    augment_kwargs: AugmentKwargs = field(default_factory=AugmentKwargs)

    just_visualize: bool = False
    distributed: bool = False
    device: str = "cuda"

    total_kimg: int = 200000
    ema_halflife_kimg: int = 500
    batch_gpu: Optional[int] = None
    loss_scaling: float = 1.0
    cudnn_benchmark: bool = True
    kimg_per_tick: int = 50
    snapshot_ticks: int = 25
    state_dump_ticks: int = 25
    log_img_ticks: int = 50
    seed: int = 2047813205
    run_dir: str = ""

    @staticmethod
    def from_yaml(file_path: str) -> 'TransdimensionalEpicConfig':
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return TransdimensionalEpicConfig(
            dataset_kwargs=DatasetKwargs(**data.get('dataset_kwargs', {})),
            distributed=data.get('distributed', False),
            device=data.get('device', "cuda"),
            data_loader_kwargs=DataLoaderKwargs(**data.get('data_loader_kwargs', {})),
            loss_kwargs=LossKwargs(**data.get('loss_kwargs', {})),
            optimizer_kwargs=OptimizerKwargs(**data.get('optimizer_kwargs', {})),
            just_visualize=data.get('just_visualize', False),
            structure_kwargs=StructureKwargs(**data.get('structure_kwargs', {})),
            sampler_kwargs=SamplerKwargs(**data.get('sampler_kwargs', {})),
            grad_conditioner_kwargs=GradConditionerKwargs(**data.get('grad_conditioner_kwargs', {})),
            encoder=NetworkKwargs(**data.get('network_kwargs', {})),
            augment_kwargs=AugmentKwargs(**data.get('augment_kwargs', {})),
            total_kimg=data.get('total_kimg', 200000),
            ema_halflife_kimg=data.get('ema_halflife_kimg', 500),
            batch_size=data.get('batch_size', 64),
            batch_gpu=data.get('batch_gpu', None),
            loss_scaling=data.get('loss_scaling', 1.0),
            cudnn_benchmark=data.get('cudnn_benchmark', True),
            kimg_per_tick=data.get('kimg_per_tick', 50),
            snapshot_ticks=data.get('snapshot_ticks', 25),
            state_dump_ticks=data.get('state_dump_ticks', 25),
            log_img_ticks=data.get('log_img_ticks', 50),
            seed=data.get('seed', 2047813205),
            run_dir=data.get('run_dir', "")
        )

    def to_yaml(self, file_path: str):
        with open(file_path, 'w') as f:
            yaml.safe_dump(asdict(self), f, default_flow_style=False)

