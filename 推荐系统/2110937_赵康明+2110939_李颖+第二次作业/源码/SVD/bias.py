
import numpy as np


"""
Statistics:
Number of users: 19835
Number of items with ratings: 455691
Total number of ratings: 5001507
Max user ID: 19834
Min user ID: 0
Max item ID: 624960
Min item ID: 0
"""

user_num = 19835
item_num = 624961
ratings_num = 5001507

def get_bias(train_data_user, train_data_item):
    """
    :param train_data_user: 用户-[物品，评分]字典
    :param train_data_item: 物品-[用户，评分]字典
    :return: 全评分均值，用户偏差, 物品偏差
    """
    miu = 0.0
    bx = np.zeros(user_num, dtype=np.float64)
    bi = np.zeros(item_num, dtype=np.float64)
    for user_id in train_data_user:
        sum = 0.0
        for item_id, score in train_data_user[user_id]:
            miu += score
            sum += score
        bx[user_id] = sum / len(train_data_user[user_id])
    miu /= ratings_num

    for item_id in train_data_item:
        sum = 0.0
        for user_id, score in train_data_item[item_id]:
            sum += score
        bi[item_id] = sum / len(train_data_item[item_id])

    bx -= miu
    bi -= miu
    return miu, bx, bi
