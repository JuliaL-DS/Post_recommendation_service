## Авторство
## Данный проект разработан и принадлежит ООО «Карпов Курсы».  
## Официальный сайт: https://karpov.courses


import os
from datetime import datetime
import hashlib
from typing import List, Tuple
import numpy as np

import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI
from loguru import logger
from schema import PostGet, Response
from sqlalchemy import create_engine

app = FastAPI()

order_1 = ['hour', 'month', 'gender', 'age', 'country', 'city', 'exp_group', 'os', 'source', 'topic', 'TextCluster', 'DistanceToCluster_0', 'DistanceToCluster_1', 
        'DistanceToCluster_2', 'DistanceToCluster_3', 'DistanceToCluster_4', 'DistanceToCluster_5', 'DistanceToCluster_6', 'DistanceToCluster_7', 'DistanceToCluster_8', 
        'DistanceToCluster_9', 'DistanceToCluster_10', 'DistanceToCluster_11', 'DistanceToCluster_12', 'DistanceToCluster_13', 'DistanceToCluster_14']  

order_2 = ['gender', 'age', 'country', 'city', 'exp_group', 'os', 'source',
        'topic', 'text_len', 'user_view_posts']  

def batch_load_sql(query: str):
    engine = create_engine("")
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=200000):
        chunks.append(chunk_dataframe)
        logger.info(f"Got chunk: {len(chunk_dataframe)}")
        break
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def load_raw_features():
    # Уникальные записи post_id, user_id, где был совершен лайк
    logger.info("loading liked posts")
    liked_posts_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        WHERE action='like'"""
    liked_posts = batch_load_sql(liked_posts_query)

    # Фичи по постам на основе tf-idf
    logger.info("loading posts features")
    posts_features = pd.read_sql(
        """SELECT * FROM public.posts_info_features_dl""",
        con=""
    )

    # Фичи по юзерам
    logger.info("loading user features")
    user_features = pd.read_sql(
        """SELECT * FROM public.user_data""", con=""
    )
    
    # Фичи по постам 
    df_posts = pd.read_sql("""SELECT * FROM public.post_text_df""", con="") 
    
    # Фичи для модели control: в фичи по юзерам добавлен столбец 'user_view_posts'
    df_users = user_features.copy()
    query_2 = """
            SELECT COUNT(user_id)
            FROM public.feed_data
            GROUP BY user_id;
        """
    df_users['user_view_posts'] = batch_load_sql(query_2)
    return [liked_posts, posts_features, user_features, df_posts, df_users]


def get_model_path(model_version: str) -> str:
    print(os.environ)
    if (
        os.environ.get("IS_LMS") == "1"
    ):  # Проверяем где выполняется код: на платформе или локально
        model_path = f"/model_{model_version}"
    else:
        model_path = (
            ""
            f"/model_{model_version}"
        )
    return model_path

def load_models(model_version: str):
    model_path = get_model_path(model_version)
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    return loaded_model

features = load_raw_features()

## Загрузка сразу двух моделей
model_control = load_models("control")
model_test = load_models("test")

## Разбиение пользователей на группы
SALT = "my_salt"

def get_user_group(id: int) -> str:
    value_str = str(id) + SALT
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    percent = value_num % 100
    if percent < 50:
        return "control"
    elif percent < 100:
        return "test"
    return "unknown"

## Рекомендации
def calculate_features(
    id: int, time: datetime, group: str
): #-> Tuple[pd.DataFrame, pd.DataFrame]:
    # Загрузка фичей по пользователям
    logger.info(f"user_id: {id}")
    logger.info("reading features")
    user_features = features[2].loc[features[2].user_id == id]
    user_features = user_features.drop("user_id", axis=1)
    
    if group == "test":        
        # Загрузка фичей по постам
        logger.info("dropping columns")
        posts_features = features[1].drop(["index"], axis=1)

        # Объединение фичей
        logger.info("zipping everything")
        add_user_features = dict(zip(user_features.columns, user_features.values[0]))
        logger.info("assigning everything")
        user_posts_features = posts_features.assign(**add_user_features)
        user_posts_features = user_posts_features.set_index("post_id")

        # Добаление информации о дате рекомендаций
        logger.info("add time info")
        user_posts_features["hour"] = time.hour
        user_posts_features["month"] = time.month
        user_posts_features = user_posts_features[order_1]
    elif group == "control":
        features[3]['text_len'] = features[3]['text'].apply(len)
        all_post_ids = features[3]['post_id'].tolist()
        row_to_duplicate = features[4].loc[features[4]['user_id'] == id]
        user_array = np.repeat(row_to_duplicate.values, len(all_post_ids), axis=0)
        user_repeated = pd.DataFrame(user_array, columns=row_to_duplicate.columns)
        extracted_column_1 = features[3]['topic']
        extracted_column_2 = features[3]['text_len']
        user_posts_features = user_repeated.assign(
            topic=pd.Series(extracted_column_1).values,
            text_len=pd.Series(extracted_column_2).values
        )
        user_posts_features = user_posts_features.drop(['user_id'], axis=1)
        user_posts_features = user_posts_features[order_2] 
    return user_posts_features

def get_recommended_feed(id: int, time: datetime, limit: int) -> Response:
    # Выбор группы пользователи
    user_group = get_user_group(id=id)
    logger.info(f"user group {user_group}")

    # Выбор нужной модели
    if user_group == "control":
        model = model_control
    elif user_group == "test":
        model = model_test
    else:
        raise ValueError("unknown group")

    # Вычисление фичей
    user_posts_features = calculate_features(
        id=id, time=time, group=user_group
    )

    # Предсказание вероятности лайкнуть пост для всех постов
    logger.info("predicting")
    predicts = model.predict_proba(user_posts_features)[:, 1]
    features[3]["predicts"] = predicts

    # Удаление записей, где пользователь ранее уже ставил лайк
    logger.info("deleting liked posts")
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = features[3][~features[3]['post_id'].isin(liked_posts)]

    # Рекомендация топ-5 по вероятности постов
    recommended_posts = filtered_.sort_values("predicts", ascending=False).iloc[:limit]

    return Response(
        recommendations=[
            PostGet(id=x.post_id, text=x.text, topic=x.topic)
            for _, x in recommended_posts.iterrows()
        ],
        exp_group=user_group
    )

@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 10) -> Response:
    return get_recommended_feed(id, time, limit)