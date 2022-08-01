# 😢<->😄 Sentiment Classifier for Japanese texts using fine-tuned BERT

Please feel free to clone this repo, and try ! 

## Content
- Fine-tuning the Japanese pre-trained BERT model for negative-positive classification.
## Pre-trained BERT model
- [cl-tohoku/bert-base-japanese-whole-word-masking](https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking)
## Dataset
- [Twitter日本語評判分析データセット](https://www.db.info.gifu-u.ac.jp/sentiment_analysis/)
## How to shape the dataset
- Undersampling original dataset so that the dataset size of both positive and negative labels is equal
## Dataset size
- Both negative and positive tweets are 8458 rows, summing up to 16916 rows.
## Dependencies
- Python 3.9.10
- Pytorch 1.10.2
- Transformers 4.16.0
## 🚀 Quick Start

 ```python
 $ git clone https://github.com/ryuryukke/bert_japanese_sentiment_classifier.git
 $ cd src
 ```
 Then, see quickstart.ipynb.



