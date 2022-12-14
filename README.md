# Sentiment Classifier for Japanese texts using fine-tuned BERT

Please feel free to clone this repo, and try !

## Content
- Fine-tuning the Japanese pre-trained BERT model for positive-negative classification.
## Pre-trained BERT model
- [cl-tohoku/bert-base-japanese-whole-word-masking](https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking)
## Dataset
- [Twitter日本語評判分析データセット](https://www.db.info.gifu-u.ac.jp/sentiment_analysis/)
## How to shape the dataset
- Undersampling original dataset so that the dataset size of both positive and negative labels are equal.
## Dataset size
- Both positive adn negative tweets are 8458 rows, summing up to 16916 rows.
## Dependencies
- python==3.9.10
- torch==1.10.2
- transformers==4.16.0
## 🚀 Quick Start

 ```python
 $ git clone https://github.com/ryuryukke/bert_japanese_sentiment_classifier.git
 $ cd src
 $ python prepare_tweets.py
 ```
 Then, see [quickstart.ipynb](https://github.com/ryuryukke/bert_japanese_sentiment_classifier/blob/master/src/quickstart.ipynb).

## 🚨 Caution
- We can't share the dataset and trained model from [Twitter日本語評判分析データセット](https://www.db.info.gifu-u.ac.jp/sentiment_analysis/) because of the license, CC-BY-ND 4.0.
    - All you need is Twitter API Key.
- If you want to use your own dataset, you have to shape that in following manner.
```python
your_own_dataset = [
    [pos_sent_1, [pos_flag(=1), neg_flag(=0)]],
    [neg_sent_2, [pos_flag(=0), neg_flag(=1)]],
    ...,
    [pos_sent_n, [pos_flag(=1), neg_flag(=0)]]
    ]
```



