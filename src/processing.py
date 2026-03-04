import json
import itertools

import torch
from torch import LongTensor
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    ClassLabel
)

from src.lemmatize_helper import construct_lemma_rule
from cobald_parser.utils import pad_sequences


ROOT_HEAD = '0'

# Sentence metadata
SENT_ID = "sent_id"
TEXT = "text"

# Fields
ID = "id"
WORD = "word"
LEMMA = "lemma"
UPOS = "upos"
XPOS = "xpos"
FEATS = "feats"
HEAD = "head"
DEPREL = "deprel"
DEPS = "deps"
MISC = "misc"

# Transformed fields
LEMMA_RULE = "lemma_rule"
JOINT_FEATS = "joint_feats"
UD_ARC_FROM = "ud_arc_from"
UD_ARC_TO = "ud_arc_to"
UD_DEPREL = "ud_deprel"


def remove_range_tokens(sentence: dict) -> dict:
    """
    Remove range tokens from a sentence.
    """
    def is_range_id(idtag: str) -> bool:
        return '-' in idtag
    
    sentence_length = len(sentence[ID])
    return {
        key: [values[i]
              for i in range(sentence_length)
              if not is_range_id(sentence[ID][i])]
        for key, values in sentence.items()
        if values is not None and isinstance(values, list)
    }


def build_counting_mask(words: list[str]) -> list:
    """Build a dummy counting mask (all zeros, no nulls)."""
    return [0] * len(words)


def transform_fields(sentence: dict) -> dict:
    """
    Transform sentence fields:
     * turn words and lemmas into lemma rules,
     * merge upos, xpos and feats into "pos-feats",
     * encode ud syntax into a single 2d matrix.
    """
    result = {}

    if LEMMA in sentence:
        result[LEMMA_RULE] = [
            construct_lemma_rule(word, lemma)
            if lemma is not None else None
            for word, lemma in zip(
                sentence[WORD],
                sentence[LEMMA],
                strict=True
            )
        ]
    
    if UPOS in sentence or XPOS in sentence or FEATS in sentence:
        result[JOINT_FEATS] = [
            f"{upos or '_'}#{xpos or '_'}#{feats or '_'}"
            if (upos is not None or xpos is not None or feats is not None) else None
            for upos, xpos, feats in zip(
                sentence[UPOS],
                sentence[XPOS],
                sentence[FEATS],
                strict=True
            )
        ]

    # Renumerate ids, so that tokens are enumerated from 0.
    # E.g. [1, 2, 3] -> [0, 1, 2].
    id2idx = {token_id: token_idx for token_idx, token_id in enumerate(sentence[ID])}

    # Basic syntax.
    if HEAD in sentence and DEPREL in sentence:
        ud_arcs_from, ud_arcs_to, ud_deprels = zip(
            *[
                (
                    id2idx[str(head_id)] if str(head_id) != ROOT_HEAD else id2idx[token_id],
                    id2idx[token_id],
                    deprel
                )
                for token_id, head_id, deprel in zip(
                    sentence[ID],
                    sentence[HEAD],
                    sentence[DEPREL],
                    strict=True
                )
                if head_id is not None
            ]
        )
        result[UD_ARC_FROM] = ud_arcs_from
        result[UD_ARC_TO] = ud_arcs_to
        result[UD_DEPREL] = ud_deprels

    return result


def extract_unique_labels(dataset, column_name) -> list[str]:
    """Extract unique labels from a specific column in the dataset."""
    all_labels = itertools.chain.from_iterable(dataset[column_name])
    unique_labels = set(all_labels)
    unique_labels.discard(None)
    return unique_labels


def build_schema_with_class_labels(tagsets: dict[str, set]) -> Features:
    """Update the schema to use ClassLabel for specified columns."""

    # Updated features schema
    features = Features({
        SENT_ID: Value("string"),
        TEXT: Value("string"),
        WORD: Sequence(Value("string"))
    })

    if LEMMA_RULE in tagsets:
        # Sort to ensure consistent ordering of labels
        lemma_rule_tagset = sorted(tagsets[LEMMA_RULE])
        features[LEMMA_RULE] = Sequence(ClassLabel(names=lemma_rule_tagset))

    if JOINT_FEATS in tagsets:
        feats_tagset = sorted(tagsets[JOINT_FEATS])
        features[JOINT_FEATS] = Sequence(ClassLabel(names=feats_tagset))

    if UD_DEPREL in tagsets:
        features[UD_ARC_FROM] = Sequence(Value('int32'))
        features[UD_ARC_TO] = Sequence(Value('int32'))
        ud_deprels_tagset = sorted(tagsets[UD_DEPREL])
        features[UD_DEPREL] = Sequence(ClassLabel(names=ud_deprels_tagset))

    if MISC in tagsets:
        misc_tagset = sorted(tagsets[MISC])
        features[MISC] = Sequence(ClassLabel(names=misc_tagset))

    return features


def replace_none_with_ignore_index(example: dict, value: int = -100) -> dict:
    """
    Replace None labels with specified value.
    """
    assert value < 0
    for name, column in example.items():
        # Skip metadata fields (they are not lists).
        if isinstance(column, list):
            example[name] = [value if item is None else item for item in column]
    return example


def transform_dataset(dataset_dict: DatasetDict) -> Dataset:
    # Transform fields.
    dataset_column_names = {
        column
        for columns in dataset_dict.column_names.values()
        for column in columns
    }
    columns_to_remove = [ID, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS]

    # Remove range tokens and transform fields.
    dataset_dict = (
        dataset_dict
        .map(remove_range_tokens)
        .map(
            transform_fields,
            remove_columns=[
                column
                for column in columns_to_remove
                if column in dataset_column_names
            ]
        )
    )
    return dataset_dict


def collate_with_padding(batches: list[dict], padding_value: int = -100) -> dict:
    def gather_column(column_name: str) -> list:
        return [batch[column_name] for batch in batches]

    def stack_padded(column_name) -> LongTensor:
        return pad_sequences(gather_column(column_name), padding_value)

    def collate_syntax(arcs_from_name: str, arcs_to_name: str, deprel_name: str) -> LongTensor:
        batch_size = len(batches)
        arcs_counts = torch.tensor([len(batch[arcs_from_name]) for batch in batches])
        batch_idxs = torch.arange(batch_size).repeat_interleave(arcs_counts)
        from_idxs = torch.concat(gather_column(arcs_from_name))
        to_idxs = torch.concat(gather_column(arcs_to_name))
        deprels = torch.concat(gather_column(deprel_name))
        return torch.stack([batch_idxs, from_idxs, to_idxs, deprels], dim=1)

    def maybe_none(labels: LongTensor) -> LongTensor | None:
        return None if labels.max() == padding_value or labels.numel() == 0 else labels
    
    result = {
        "words": gather_column(WORD),
        "sent_ids": gather_column(SENT_ID),
        "texts": gather_column(TEXT)
    }

    columns = {column for batch in batches for column in batch}
    if LEMMA_RULE in columns:
        lemma_rules_batched = stack_padded(LEMMA_RULE)
        result["lemma_rules"] = maybe_none(lemma_rules_batched)

    if JOINT_FEATS in columns:
        joint_feats_batched = stack_padded(JOINT_FEATS)
        result["joint_feats"] = maybe_none(joint_feats_batched)

    if UD_DEPREL in columns:
        deps_ud_batched = collate_syntax(UD_ARC_FROM, UD_ARC_TO, UD_DEPREL)
        result["deps_ud"] = maybe_none(deps_ud_batched)

    if MISC in columns:
        miscs_batched = stack_padded(MISC)
        result["miscs"] = maybe_none(miscs_batched)

    return result