# Text2Class
Build multi-class text classifiers using state-of-the-art pre-trained contextualized language models, e.g. BERT. Only a few hundred samples per class are necessary to get started.

## Background

This project is based on our study: [Transfer Learning Robustness in Multi-Class Categorization by Fine-Tuning Pre-Trained Contextualized Language Models](https://arxiv.org/abs/1909.03564).

### Citation

To cite this work, use the following BibTeX citation.

```
@article{transfer2019multiclass,
  title={Transfer Learning Robustness in Multi-Class Categorization by Fine-Tuning Pre-Trained Contextualized Language Models},
  author={Liu, Xinyi and Wangperawong, Artit},
  journal={arXiv preprint arXiv:1909.03564},
  year={2019}
}
```

## Installation
```
pip install text2class
```

## Example usage

### Create a dataframe with two columns, such as 'text' and 'label'. No text pre-processing is necessary.
```
import pandas as pd
from text2class.text_classifier import TextClassifier

df = pd.read_csv("data.csv")

train = df.sample(frac=0.9,random_state=200)
test = df.drop(train.index)

cls = TextClassifier(
	num_labels=3,
	data_column="text",
	label_column="label",
	max_seq_length=128
)

cls.fit(train)

predictions = cls.predict(test["text"])
```

## Advanced usage

### Model type
The default model is an uncased Bidirectional Encoder Representations from Transformers (BERT) consisting of 12 transformer layers, 12 self-attention heads per layer, and a hidden size of 768. Below are all models currently supported that you can specify with `hub_module_handle`. We expect that more will be added in the future. For more information, see [BERT's GitHub](https://github.com/google-research/bert).
```
https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1
https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1
https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1
https://tfhub.dev/google/bert_cased_L-24_H-1024_A-16/1
https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1
https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1

cls = TextClassifier(
	num_labels=3,
	data_column="text",
	label_column="label",
	max_seq_length=128,
	hub_module_handle="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
)
```

## Contributing
Text2Class is an open-source project founded and maintained to better serve the machine learning and data science community. Please feel free to submit pull requests to contribute to the project. By participating, you are expected to adhere to Text2Class's [code of conduct](CODE_OF_CONDUCT.md).

## Questions?
For questions or help using Text2Class, please submit a GitHub issue.
