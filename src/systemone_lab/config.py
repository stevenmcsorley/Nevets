from dataclasses import dataclass, asdict, field
from pathlib import Path
import yaml

@dataclass
class ModelConfig:
    name: str = "s1-35m"
    vocab_size: int = 16000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    d_ff: int = 1408
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0
    pointer_dim: int = 256

    decision_head: dict = field(default_factory=lambda: {"scorer": "scaled_dot", "temperature": 10.0})
    # Optional recurrent layout: {"type": "looped", "prelude": P, "core": C, "coda": D, "iters": K}
    # with P + C + D == n_layers. The core is weight-tied and re-applied K times with input injection.
    arch: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self):
        return asdict(self)
