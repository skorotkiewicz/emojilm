from dataclasses import dataclass, asdict


@dataclass
class Config:
    vocab: int = 4096
    ctx: int = 256
    layers: int = 8
    dim: int = 320
    heads: int = 5
    ffn: int = 1280

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config):
        return cls(**config)
