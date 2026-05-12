from transformers import PreTrainedTokenizer
from typing import List, Optional, Dict


class NucleotideTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, **kwargs):

        kwargs.update({
            'pad_token': '[PAD]',
            'unk_token': '[UNK]',
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'mask_token': '[MASK]'
        })

        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f if line.strip()]

        self.vocab = {v: i for i, v in enumerate(vocab)}
        self.ids_to_tokens = {i: v for i, v in enumerate(vocab)}

        for token in [kwargs['pad_token'], kwargs['unk_token'],
                      kwargs['cls_token'], kwargs['sep_token'],
                      kwargs['mask_token']]:
            if token not in self.vocab:
                raise ValueError(f"specila token {token} not in vocab.txt")

        super().__init__(**kwargs)

    def _tokenize(self, text: str) -> List[str]:
        tokens = []
        i = 0
        while i < len(text):

            found = False
            for special in ['[MASK]', '[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                if text.startswith(special, i):
                    tokens.append(special)
                    i += len(special)
                    found = True
                    break
            if found:
                continue

            tokens.append(text[i])
            i += 1
        return tokens

    def _convert_token_to_id(self, token: str) -> int:

        return self.vocab.get(token, self.vocab.get("[UNK]"))

    def _convert_id_to_token(self, index: int) -> str:

        return self.ids_to_tokens.get(index, "[UNK]")

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):

        import os
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.txt"
        )

        with open(vocab_file, 'w', encoding='utf-8') as f:
            for token in self.vocab.keys():
                f.write(token + '\n')

        return (vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]

    def get_special_tokens_mask(self, token_ids, already_has_special_tokens=False):

        if already_has_special_tokens:
            return [1 if token in [self.cls_token_id, self.sep_token_id] else 0 for token in token_ids]
        return [0] * len(token_ids)

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
