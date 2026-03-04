import torch
from torch import Tensor


def pad_sequences(sequences: list[Tensor], padding_value: int) -> Tensor:
    """
    Stack 1d tensors (sequences) into a single 2d tensor so that each sequence is padded on the
    right.
    """
    return torch.nn.utils.rnn.pad_sequence(sequences, padding_value=padding_value, batch_first=True)


def _build_condition_mask(sentences: list[list[str]], condition_fn: callable, device) -> Tensor:
    masks = [
        torch.tensor([condition_fn(word) for word in sentence], dtype=bool, device=device)
        for sentence in sentences
    ]
    return pad_sequences(masks, padding_value=False)

def build_padding_mask(sentences: list[list[str]], device) -> Tensor:
    return _build_condition_mask(sentences, condition_fn=lambda word: True, device=device)

def pairwise_mask(masks1d: Tensor) -> Tensor:
    """
    Calculate an outer product of a mask, i.e. masks2d[:, i, j] = masks1d[:, i] & masks1d[:, j].
    """
    return masks1d[:, None, :] & masks1d[:, :, None]


# Credits: https://docs.allennlp.org/main/api/nn/util/#replace_masked_values
def replace_masked_values(tensor: Tensor, mask: Tensor, replace_with: float):
    """
    Replace all masked values in tensor with `replace_with`.
    """
    assert tensor.dim() == mask.dim(), "tensor.dim() of {tensor.dim()} != mask.dim() of {mask.dim()}"
    tensor.masked_fill_(~mask, replace_with)


