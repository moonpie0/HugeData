import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from convert_data import *
from bias import  *
import time
# 加载数据
def load_data(train_path='./data/train.txt', test_path='./data/test.txt'):
    
    train_user_data,train_item_data = get_train_data(train_path)
    _,bx,bi = get_bias(train_user_data,train_item_data)
    train_data, valid_data = split_data(train_user_data)
    train_data=train_user_data
    #valid_data=get_valid_data('validation_data.txt')
    test_data = get_test_data(test_path)
    train_items=get_train_items(train_data)
    return bx, bi, train_data, valid_data, test_data, train_items

# 计算全局平均分
def calculate_global_mean(train_user_data):
    score_sum, count = 0.0, 0
    for items in train_user_data.values():
        for item_id, score in items:
            score_sum += score
            count += 1
    return score_sum / count if count > 0 else 0

# 预测评分
def predict(user_id, item_id, bx, bi, P, Q, global_mean):
    pre_score = (global_mean + 
                 bx[user_id] + 
                 bi[item_id] + 
                 np.dot(P[:, user_id], Q[:, item_id]))
    return pre_score

# 计算损失
def calculate_loss(data, bx, bi, P, Q, global_mean):
    loss, count = 0.0, 0
    for user_id, items in data.items():
        for item_id, true_score in items:
            pred_score = predict(user_id, item_id, bx, bi, P, Q, global_mean)
            loss += ((true_score - pred_score)*10) ** 2
            count += 1
    return loss / count if count > 0 else 0

# 计算RMSE
def calculate_rmse(data, bx, bi, P, Q, global_mean):
    loss, count = 0.0, 0
    for user_id, items in data.items():
        for item_id, true_score in items:
            pred_score = predict(user_id, item_id, bx, bi, P, Q, global_mean)
            loss += ((true_score - pred_score)*10) ** 2
            count += 1
    return np.sqrt(loss / count) if count > 0 else float('inf')

# 训练模型
def train(bx, bi, train_data, valid_data, factor=50, lr=5e-3, lambda1=1e-2, lambda2=1e-2, lambda3=1e-2, lambda4=1e-2, epochs=10):
    global_mean = calculate_global_mean(train_data)
    Q = np.random.normal(0, 0.1, (factor, len(bi)))
    P = np.random.normal(0, 0.1, (factor, len(bx)))

    for epoch in range(epochs):
        train_loss = train_one_epoch(bx, bi, train_data, P, Q, global_mean, lr, lambda1, lambda2, lambda3, lambda4)
        #valid_loss = calculate_loss(valid_data, bx, bi, P, Q, global_mean)
        train_rmse = calculate_rmse(train_data, bx, bi, P, Q, global_mean)
        valid_rmse = calculate_rmse(valid_data, bx, bi, P, Q, global_mean)
        print(f'Epoch {epoch + 1} train loss: {train_loss:.6f} train RMSE: {train_rmse:.6f} valid RMSE: {valid_rmse:.6f}')
    return P, Q

# 单个训练周期
def train_one_epoch(bx, bi, train_data, P, Q, global_mean, lr, lambda1, lambda2, lambda3, lambda4):
    train_loss, count = 0.0, 0
    for user_id, items in tqdm(train_data.items()):
        for item_id, true_score in items:
            pred_score = predict(user_id, item_id, bx, bi, P, Q, global_mean)
            error = true_score - pred_score

            # 更新参数
            bx[user_id] += lr * (error - lambda3 * bx[user_id])
            bi[item_id] += lr * (error - lambda4 * bi[item_id])
            P[:, user_id] += lr * (error * Q[:, item_id] - lambda1 * P[:, user_id])
            Q[:, item_id] += lr * (error * P[:, user_id] - lambda2 * Q[:, item_id])

            train_loss += error ** 2
            count += 1
    return train_loss / count if count > 0 else 0

# 获取训练集中的所有物品ID
def get_train_items(train_data):
    train_items = set()
    for user_id, items in train_data.items():
        for item_id, _ in items:
            train_items.add(item_id)
    return train_items

# 检查预测物品是否存在于训练集中
def item_exists_in_train(item_id, train_items):
    return item_id in train_items
# 测试模型
def write_result(predict_score, write_path):
    with open(write_path, 'w') as f:
        for user_id, items in predict_score.items():
            f.write(f'{user_id}|6\n')
            for item_id, score in items:
                f.write(f'{item_id} {score}\n')
                
def test(bx, bi, test_data, train_items, P, Q, global_mean, write_path='./result/svd_result.txt'):
    predict_score = defaultdict(list)  # 保存预测评分的字典
    for user_id, item_ids in test_data.items():
        for item_id in item_ids:
            if item_id not in train_items:
                pred_score = global_mean * 10
            else:
                pred_score = predict(user_id, item_id, bx, bi, P, Q, global_mean) * 10
                
            # 限制评分范围
            if pred_score > 100.0:
                pred_score = 100.0
            elif pred_score < 0.0:
                pred_score = 0.0
            
            # 保存预测评分到字典中
            predict_score[user_id].append((item_id, pred_score))
    
    # 写入结果到文件
    write_result(predict_score, write_path)
    print('Testing completed and results written.')

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f'{hours}h {minutes}m {seconds}s'

if __name__ == '__main__':
    # 主函数
    start_time = time.time()  # 记录开始时间
    bx, bi, train_data, valid_data, test_data, train_items = load_data()
    P, Q = train(bx, bi, train_data, valid_data)
    global_mean = calculate_global_mean(train_data)
    test(bx, bi, test_data, train_items,P, Q, global_mean)
    rmse = calculate_rmse(train_data, bx, bi, P, Q, global_mean)
    print(f'RMSE: {rmse:.6f}')

    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time  # 计算执行时间

    # 输出执行时间，格式为小时、分钟、秒
    formatted_time = format_time(execution_time)
    print(f'Total Execution Time: {formatted_time}')

