import os
from typing import override

from torch.optim import AdamW
from transformers import Trainer
from transformers.modelcard import parse_log_history
from huggingface_hub import ModelCard, ModelCardData, EvalResult


MODELCARD_TEMPLATE = """
---
{{ card_data }}
---

# Model Card for {{ model_name }}

A transformer-based multihead parser for CoBaLD annotation.

This model parses a pre-tokenized CoNLL-U text and jointly labels each token with:
* Grammatical tags (lemma, UPOS, XPOS, morphological features),
* Syntactic tags (basic Universal Dependencies).

## Model Sources

- **Repository:** https://github.com/CobaldAnnotation/CobaldParser
- **Paper:** https://dialogue-conf.org/wp-content/uploads/2025/04/BaiukIBaiukAPetrovaM.009.pdf
- **Demo:** [coming soon]

## Citation

```
@inproceedings{baiuk2025cobald,
  title={CoBaLD Parser: Joint Morphosyntactic and Semantic Annotation},
  author={Baiuk, Ilia and Baiuk, Alexandra and Petrova, Maria},
  booktitle={Proceedings of the International Conference "Dialogue"},
  volume={I},
  year={2025}
}
```
"""


class CustomTrainer(Trainer):
    @override
    def create_model_card(self, **kwargs):
        """Create custom model card."""

        dataset = self.eval_dataset
        organization, model_name = self.hub_model_id.split('/')
        hub_dataset_id = f"{organization}/{dataset.info.dataset_name}"

        _, _, eval_results_plain = parse_log_history(self.state.log_history)

        eval_results = []
        for metric_name, metric_type in (
            ('Lemma F1', 'f1'),
            ('Morphology F1', 'f1'),
            ('Ud Jaccard', 'accuracy'),
            ('Miscs F1', 'f1')
        ):
            if metric_name in eval_results_plain:
                eval_result = EvalResult(
                    task_type='token-classification',
                    dataset_type=hub_dataset_id,
                    dataset_name=dataset.info.dataset_name,
                    dataset_split='validation',
                    metric_name=metric_name,
                    metric_type=metric_type,
                    metric_value=eval_results_plain[metric_name]
                )
                eval_results.append(eval_result)

        card = ModelCard.from_template(
            card_data=ModelCardData(
                base_model=self.model.config.encoder_model_name,
                datasets=hub_dataset_id,
                language=dataset.info.config_name,
                eval_results=eval_results,
                library_name='transformers',
                license='gpl-3.0',
                metrics=['accuracy', 'f1'],
                model_name=self.hub_model_id,
                pipeline_tag='token-classification',
                tags=['pytorch']
            ),
            template_str=MODELCARD_TEMPLATE,
            model_name=model_name
        )
        model_card_filepath = os.path.join(self.args.output_dir, "README.md")
        card.save(model_card_filepath)

    @override
    def create_optimizer(self):
        # Implement discriminative‐finetuning.
        # NOTE: it breaks multiple CLI features like `--fp16` and `--fsdp`, but
        # we don't need them so far anyway...

        if self.optimizer is not None:
            return self.optimizer
        
        base_lr = self.args.learning_rate
        encoder_lr = base_lr / 5
        decay = self.args.weight_decay
        layer_decay = 0.9
        optimizer_grouped_parameters = []

        # Add classifier with the base LR
        optimizer_grouped_parameters.append({
            "params": self.model.classifiers.parameters(),
            "lr": base_lr,
            "weight_decay": decay
        })
        
        # Per‐layer parameter groups with decaying LR
        layers = self.model.encoder.get_transformer_layers()
        for idx, layer in enumerate(layers):
            lr = encoder_lr * (layer_decay ** (len(layers) - idx - 1))
            optimizer_grouped_parameters.append({
                "params": layer.parameters(),
                "lr": lr,
                "weight_decay": decay
            })

        # Add embeddings with the smallest LR
        embeddings = self.model.encoder.get_embeddings_layer()
        smallest_lr = encoder_lr * (layer_decay ** len(layers))
        optimizer_grouped_parameters.append({
            "params": embeddings.parameters(),
            "lr": smallest_lr,
            "weight_decay": decay
        })

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon
        )
        return self.optimizer