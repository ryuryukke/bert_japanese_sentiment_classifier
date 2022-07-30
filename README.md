## Sentiment Classifier for Japanese texts using fine-tuned BERT
- 概要
    - 日本語事前学習済みBERTモデルをポジネガ判定のためにFine-tuningしたモデル
- 使用モデル
    - [東北大の日本語事前学習済みBERTモデル](https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking)
- データセット
    - [twitterポジネガデータセット](https://www.db.info.gifu-u.ac.jp/sentiment_analysis/)
- データ収集
    - こちらのコードでStatusIdからTwitterAPIで取得
- データ整形
    - ポジネガ両ラベルのデータサイズが等しくなるようにアンダーサンプリング
- データサイズ
    - ポジ、ネガともに8458ツイートで合わせて16916ツイート
 ## Quick Start
 ```python
 
 ```