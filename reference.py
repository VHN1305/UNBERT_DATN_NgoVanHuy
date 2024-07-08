import torch
import argparse
import pandas as pd
import os
from src.model import UNBERT

news_df = pd.read_csv("data/small/dev/news.tsv", sep="\t", header=None)
news_df.columns = ["news_id", "category", "sub_category", "title", "abstract", "url", "title_entities", "abstract_entities"]

behaviors_df = pd.read_csv("data/small/dev/behaviors.tsv", sep="\t", header=None)
behaviors_df.columns = ["impression_id", "user_id", "time", "history", "impressions"]

dev_score = pd.read_csv("final_dev_score_rank.tsv", sep="\t")

def get_news_info(news_id):
    try:
        news_category = news_df[news_df["news_id"] == news_id]["category"].values[0]
    except:
        news_category = "nan"
    try:
        news_sub_category = news_df[news_df["news_id"] == news_id]["sub_category"].values[0]
    except:
        news_sub_category = "nan"
    try:
        news_title = news_df[news_df["news_id"] == news_id]["title"].values[0]
    except:
        news_title = "nan"
    try:
        news_abstract = news_df[news_df["news_id"] == news_id]["abstract"].values[0]
    except:
        news_abstract = "nan"
    try:
        news_url = news_df[news_df["news_id"] == news_id]["url"].values[0]
    except:
        news_url = "nan"
    return news_id, news_category, news_sub_category, news_title, news_abstract, news_url

def get_recommend_news_rank_list(impression_id):
    candidate_news = behaviors_df[behaviors_df["impression_id"] == impression_id]["impressions"].values[0].split()
    candidate_news = [news.split("-")[0] for news in candidate_news]
    ranks_news = dev_score[dev_score["impression_id"] == impression_id-1]["rank"].values
    news_rank_pairs = list(zip(candidate_news, ranks_news))
    sorted_news_rank_pairs = sorted(news_rank_pairs, key=lambda x: x[1])
    sorted_news = [news for news, rank in sorted_news_rank_pairs]
    return sorted_news

def get_history_news_list(impression_id):
    history_news = behaviors_df[behaviors_df["impression_id"] == impression_id]["history"].values[0].split()
    return history_news

def get_impression_id(user_id):
    impression_id = behaviors_df[behaviors_df["user_id"] == user_id]["impression_id"].values[0]
    return impression_id

if __name__ == '__main__':
    # impression = get_impression_id("U23420")
    # print(impression)
    # history_list = get_history_news_list(130)
    # print(f"History News List: {history_list}")
    # for news_id in history_list:
    #     news_info = get_news_info(news_id)
    #     print(f"News ID: {news_info[0]}")
    #     print(f"Category: {news_info[1]}")
    #     print(f"Sub Category: {news_info[2]}")
    #     print(f"Title: {news_info[3]}")
    #     print(f"Abstract: {news_info[4]}")
    #     print(f"URL: {news_info[5]}")
    #     print("="*50)
    # print("="*100)
    rank_list = get_recommend_news_rank_list(2)
    print(f"Recommend News List: {rank_list}")
    for news_id in rank_list:
        news_info = get_news_info(news_id)
        print(f"News ID: {news_info[0]}")
        print(f"Category: {news_info[1]}")
        print(f"Sub Category: {news_info[2]}")
        print(f"Title: {news_info[3]}")
        print(f"Abstract: {news_info[4]}")
        print(f"URL: {news_info[5]}")
        print("="*50)
