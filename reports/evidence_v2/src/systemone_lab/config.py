from dataclasses import dataclass, asdict
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

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls(**yaml.safe_load(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)
