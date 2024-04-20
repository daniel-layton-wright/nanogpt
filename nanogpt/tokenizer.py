import numpy as np
import torch


class CharacterTokenizer:
    def __init__(self, chars=None, data_filename=None):
        if chars is not None:
            self.chars = chars
        elif data_filename is not None:
            self.chars = self.get_characters(data_filename)

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write(''.join(self.chars))

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'r') as f:
            chars = f.read()

        return cls(chars=chars)

    @staticmethod
    def get_characters(input_filename):
        with open(input_filename, 'r') as f:
            data = f.read()

        print(f'Length of input data in characters: {len(data):,}')

        chars = sorted(list(set(data)))
        vocab_size = len(chars)

        print(f'Number of unique characters: {vocab_size:,}')
        print(f'Characters: {chars}')

        return chars

    def encode(self, text) -> torch.Tensor:
        return torch.tensor([self.chars.index(c) for c in text], dtype=torch.long)

    def encode_to_file(self, text, filename):
        encoded = self.encode(text)
        torch.save(encoded, filename)

    def encode_from_file_and_save(self, input_filename, output_filename):
        with open(input_filename, 'r') as f:
            text = f.read()

        self.encode_to_file(text, output_filename)

    def decode(self, tokens):
        return ''.join([self.chars[t] for t in tokens])
