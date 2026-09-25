from pathlib import Path
from typing import Iterable

SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>", "<unk>",
    "<state>", "</state>", "<q>", "</q>",
    "<opt>", "</opt>", "<decide>", "<noul>", "<choice>", "<score>"
]

class LabTokenizer:
    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(path))
        self.ids = {t: self.tok.token_to_id(t) for t in SPECIAL_TOKENS}
        if any(v is None for v in self.ids.values()):
            missing = [k for k,v in self.ids.items() if v is None]
            raise ValueError(f"Tokenizer missing required special tokens: {missing}")

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    def id(self, token: str) -> int:
        return self.ids[token]


def train_tokenizer(text_iter: Iterable[str], out_path: str | Path, vocab_size: int = 16000):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.trainers import BpeTrainer

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, min_frequency=2)
    tok.train_from_iterator(text_iter, trainer=trainer)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok
