from torch import nn
from torch import LongTensor
from transformers import PreTrainedModel

from .configuration import CobaldParserConfig
from .encoder import WordTransformerEncoder
from .mlp_classifier import MlpClassifier
from .dependency_classifier import DependencyClassifier
from .utils import build_padding_mask


class CobaldParser(PreTrainedModel):
    """Morpho-Syntactic Parser."""

    config_class = CobaldParserConfig

    def __init__(self, config: CobaldParserConfig):
        super().__init__(config)

        self.encoder = WordTransformerEncoder(
            model_name=config.encoder_model_name
        )
        embedding_size = self.encoder.get_embedding_size()

        self.classifiers = nn.ModuleDict()
        if "lemma_rule" in config.vocabulary:
            self.classifiers["lemma_rule"] = MlpClassifier(
                input_size=embedding_size,
                hidden_size=config.lemma_classifier_hidden_size,
                n_classes=len(config.vocabulary["lemma_rule"]),
                activation=config.activation,
                dropout=config.dropout
            )
        if "joint_feats" in config.vocabulary:
            self.classifiers["joint_feats"] = MlpClassifier(
                input_size=embedding_size,
                hidden_size=config.morphology_classifier_hidden_size,
                n_classes=len(config.vocabulary["joint_feats"]),
                activation=config.activation,
                dropout=config.dropout
            )
        if "ud_deprel" in config.vocabulary:
            self.classifiers["syntax"] = DependencyClassifier(
                input_size=embedding_size,
                hidden_size=config.dependency_classifier_hidden_size,
                n_rels=len(config.vocabulary["ud_deprel"]),
                activation=config.activation,
                dropout=config.dropout
            )
        if "misc" in config.vocabulary:
            self.classifiers["misc"] = MlpClassifier(
                input_size=embedding_size,
                hidden_size=config.misc_classifier_hidden_size,
                n_classes=len(config.vocabulary["misc"]),
                activation=config.activation,
                dropout=config.dropout
            )

    def forward(
        self,
        words: list[list[str]],
        lemma_rules: LongTensor = None,
        joint_feats: LongTensor = None,
        deps_ud: LongTensor = None,
        miscs: LongTensor = None,
        sent_ids: list[str] = None,
        texts: list[str] = None,
        inference_mode: bool = False
    ) -> dict:
        output = {}

        # Encode words.
        # [batch_size, seq_len, embedding_size]
        embeddings = self.encoder(words)

        output["words"] = words
        output["loss"] = 0.0

        # Predict lemmas and morphological features.
        if "lemma_rule" in self.classifiers:
            lemma_output = self.classifiers["lemma_rule"](embeddings, lemma_rules)
            output["lemma_rules"] = lemma_output['preds']
            output["loss"] += lemma_output['loss']

        if "joint_feats" in self.classifiers:
            joint_feats_output = self.classifiers["joint_feats"](embeddings, joint_feats)
            output["joint_feats"] = joint_feats_output['preds']
            output["loss"] += joint_feats_output['loss']

        # Predict syntax.
        if "syntax" in self.classifiers:
            padding_mask = build_padding_mask(words, self.device)
            deps_output = self.classifiers["syntax"](
                embeddings,
                deps_ud,
                padding_mask
            )
            output["deps_ud"] = deps_output['preds']
            output["loss"] += deps_output['loss']

        # Predict miscellaneous features.
        if "misc" in self.classifiers:
            misc_output = self.classifiers["misc"](embeddings, miscs)
            output["miscs"] = misc_output['preds']
            output["loss"] += misc_output['loss']

        return output