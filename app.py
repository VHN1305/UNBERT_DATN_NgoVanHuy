from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
from reference import get_news_info, get_recommend_news_rank_list, get_history_news_list


app = Flask(__name__)

news_df = pd.read_csv("data/small/dev/news.tsv", sep="\t", header=None)
news_df.columns = ["news_id", "category", "sub_category", "title",
                   "abstract", "url", "title_entities", "abstract_entities"]
# fill nan values
news_df.fillna("nan", inplace=True)

behaviors_df = pd.read_csv("data/small/dev/behaviors.tsv", sep="\t", header=None)
behaviors_df.columns = ["impression_id",
                        "user_id", "time", "history", "impressions"]
behaviors_df.fillna("nan", inplace=True)
dev_score = pd.read_csv("final_dev_score_rank.tsv", sep="\t")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/get_news/<news_id>', methods=['GET'])
def get_news_info(news_id):
    try:
        news_category = news_df[news_df["news_id"]
                                == news_id]["category"].values[0]
    except:
        news_category = "nan"
    try:
        news_sub_category = news_df[news_df["news_id"]
                                    == news_id]["sub_category"].values[0]
    except:
        news_sub_category = "nan"
    try:
        news_title = news_df[news_df["news_id"] == news_id]["title"].values[0]
    except:
        news_title = "nan"
    try:
        news_abstract = news_df[news_df["news_id"]
                                == news_id]["abstract"].values[0]
    except:
        news_abstract = "nan"
    try:
        news_url = news_df[news_df["news_id"] == news_id]["url"].values[0]
    except:
        news_url = "nan"

    return jsonify({
        "news_category": news_category,
        "news_sub_category": news_sub_category,
        "news_title": news_title,
        "news_abstract": news_abstract,
        "news_url": news_url

    })


@app.route('/get_user_list/<impression_id>', methods=['GET'])
def get_users_info(impression_id):
    impression_id = int(impression_id)
    try:
        history_news = behaviors_df[behaviors_df["impression_id"]
                                    == impression_id]["history"].values[0].split()
    except:
        history_news = 'nan'
    data = []
    for news_id in history_news:

        try:
            news_category = news_df[news_df["news_id"]
                                    == news_id]["category"].values[0]
        except:
            news_category = "nan"
        try:
            news_sub_category = news_df[news_df["news_id"]
                                        == news_id]["sub_category"].values[0]
        except:
            news_sub_category = "nan"
        try:
            news_title = news_df[news_df["news_id"]
                                 == news_id]["title"].values[0]
        except:
            news_title = "nan"
        try:
            news_abstract = news_df[news_df["news_id"]
                                    == news_id]["abstract"].values[0]
        except:
            news_abstract = "nan"
        try:
            news_url = news_df[news_df["news_id"] == news_id]["url"].values[0]
        except:
            news_url = "nan"
        data.append(
            {
                "news_category": news_category,
                "news_sub_category": news_sub_category,
                "news_title": news_title,
                "news_abstract": news_abstract,
                "news_url": news_url
            }
        )
    return data

def get_recommend_news_rank_list(impression_id):
    candidate_news = behaviors_df[behaviors_df["impression_id"] == impression_id]["impressions"].values[0].split()
    candidate_news = [news.split("-")[0] for news in candidate_news]
    ranks_news = dev_score[dev_score["impression_id"] == impression_id-1]["rank"].values
    news_rank_pairs = list(zip(candidate_news, ranks_news))
    sorted_news_rank_pairs = sorted(news_rank_pairs, key=lambda x: x[1])
    sorted_news = [news for news, rank in sorted_news_rank_pairs]
    return sorted_news

@app.route('/get_recommend_list/<impression_id>', methods=['GET'])
def get_recommend_info(impression_id):
    impression_id = int(impression_id)
    try:
        recommend_news = get_recommend_news_rank_list(impression_id)
    except:
        recommend_news = 'nan'
 
    data = []
    for news_id in recommend_news:
        try:
            news_category = news_df[news_df["news_id"]
                                    == news_id]["category"].values[0]
        except:
            news_category = "nan"

        try:
            news_sub_category = news_df[news_df["news_id"]
                                        == news_id]["sub_category"].values[0]
        except:
            news_sub_category = "nan"

        try:
            news_title = news_df[news_df["news_id"]
                                 == news_id]["title"].values[0]
        except:
            news_title = "nan"

        try:
            news_abstract = news_df[news_df["news_id"]
                                    == news_id]["abstract"].values[0]
        except:
            news_abstract = "kdskds"

        try:
            news_url = news_df[news_df["news_id"] == news_id]["url"].values[0]
        except:
            news_url = "nan"
        data.append(
            {
                "news_category": news_category,
                "news_sub_category": news_sub_category,
                "news_title": news_title,
                "news_abstract": news_abstract,
                "news_url": news_url
            }
        )
    return data



if __name__ == '__main__':
    app.run(debug=True)
