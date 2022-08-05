import pandas as pd
import tweepy
import pickle
from tqdm import tqdm
import time
import random
from typing import Optional

random.seed(42)

### Add Your Own Twitter API Config ###
consumer_key = "xxxxxxxxxxxxxxxxxx"
consumer_secret = "xxxxxxxxxxxxxxxxxx"
access_token = "xxxxxxxxxxxxxxxxxx"
access_token_secret = "xxxxxxxxxxxxxxxxxx"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


def extract_pos_neg_data_from_tweets_open_csv():
    tweets_df = pd.read_csv("../data/tweets_open.csv")
    # Original tweets_open.csv doesn't include column row, so it needs an operation like this.
    # Add first row in tweets_open.csv, then add column row
    df_add = pd.DataFrame([10025, 10000, 522407718091366400, 0, 0, 1, 1, 0]).transpose()
    df_add = df_add.set_axis(tweets_df.columns.to_list(), axis=1)
    clean_df = pd.concat([df_add, tweets_df], axis=0).reset_index(drop=True)
    clean_df = clean_df.set_axis(["_", "_", "tweet-ID", "_", "pos", "neg", "_", "_"], axis=1)
    _tweets_df = clean_df[["tweet-ID", "pos", "neg"]]
    pos_neg_df = _tweets_df[~(_tweets_df["pos"] == 0) | ~(_tweets_df["neg"] == 0)]
    pos_df = pos_neg_df[pos_neg_df["pos"] == 1]
    neg_df = pos_neg_df[pos_neg_df["neg"] == 1]
    return pos_df, neg_df


def fetch_tweets(tweet_ids: list[int]) -> list[str]:
    tweets = []
    for tweet_id in tqdm(tweet_ids):
        try:
            status = api.get_status(tweet_id)
        except:
            pass
        else:
            tweets.append(status.text)
        finally:
            time.sleep(0.5)
    return tweets


def dump_pickled_obj(path: str, obj: Optional) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


if __name__ == "__main__":
    pos_df, neg_df = extract_pos_neg_data_from_tweets_open_csv()

    pos_tweets = fetch_tweets(pos_df["tweet-ID"][:3])
    neg_tweets = fetch_tweets(neg_df["tweet-ID"][:3])

    # 改行文字を削除
    preprocessed_pos_tweets = [tweet.replace("\n", " ").replace("  ", " ") for tweet in pos_tweets]
    preprocessed_neg_tweets = [tweet.replace("\n", " ").replace("  ", " ") for tweet in neg_tweets]

    labeled_pos_tweets = [[pos_tweet, [1, 0]] for pos_tweet in preprocessed_pos_tweets]
    labeled_neg_tweets = [[neg_tweet, [0, 1]] for neg_tweet in preprocessed_neg_tweets]

    # 不均衡データなので、negデータの数を減らす、undersampling
    random.shuffle(labeled_neg_tweets)
    sampled_labeled_neg_tweets = labeled_neg_tweets[: len(labeled_pos_tweets)]
    labeled_neg_pos_tweets = sampled_labeled_neg_tweets + labeled_pos_tweets
    random.shuffle(labeled_neg_pos_tweets)

    dump_pickled_obj("../data/tweets.pkl", labeled_neg_pos_tweets)
