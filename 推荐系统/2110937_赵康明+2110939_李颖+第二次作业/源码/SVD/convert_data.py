import numpy as np
from collections import defaultdict

# 文件路径定义
train_path = './data/train.txt'
test_path = './data/test.txt'
attribute_path = './data/itemAttribute.txt'

# 读取训练数据
def get_train_data(train_path):
    data_user, data_item = defaultdict(list), defaultdict(list)
    with open(train_path, 'r') as f:
        while (line := f.readline()) != '':
            user_id, num = map(int, line.strip().split('|'))
            for _ in range(num):
                line = f.readline()
                item_id, score = line.strip().split()
                item_id, score = int(item_id), float(score)
                # 把0-100的得分映射到0-10
                score = score / 10
                data_user[user_id].append([item_id, score])
                data_item[item_id].append([user_id, score])
    return data_user, data_item

# 读取物品属性数据
def get_attribute_data(attribute_path='./data/itemAttribute.txt'):
    attrs = defaultdict(list)
    with open(attribute_path, 'r') as f:
        while (line := f.readline()) != '':
            item_id, attr1, attr2 = line.strip().split('|')
            attr1 = 0 if attr1 == 'None' else int(attr1)
            attr2 = 0 if attr2 == 'None' else int(attr2)
            item_id = int(item_id)
            attrs[item_id].extend([attr1, attr2])
    return attrs
def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)
def normalize_attributes(attributes):
    normalized_attributes = {}
    for item_id, attr_values in attributes.items():
        # 将属性值拆分为两个属性
        attribute1 = attr_values[0::2]
        attribute2 = attr_values[1::2]

        min_attr1 = min(attribute1)
        max_attr1 = max(attribute1)
        min_attr2 = min(attribute2)
        max_attr2 = max(attribute2)
        
        # 检查属性值范围是否为零，如果是，则设置范围为一个很小的正数
        if min_attr1 == max_attr1:
            min_attr1 -= 1e-6
            max_attr1 += 1e-6
        if min_attr2 == max_attr2:
            min_attr2 -= 1e-6
            max_attr2 += 1e-6
        
        normalized_attr1 = [normalize(value, min_attr1, max_attr1) for value in attribute1]
        normalized_attr2 = [normalize(value, min_attr2, max_attr2) for value in attribute2]
        normalized_attributes[item_id] = (normalized_attr1, normalized_attr2)
    return normalized_attributes



# 根据属性划分数据
def get_data_by_attr(data_user, attrs):
    data1, data2 = defaultdict(list), defaultdict(list)
    for user_id, items in data_user.items():
        for item_id, score in items:
            if attrs[item_id]:
                if attrs[item_id][0] == 1:
                    data1[user_id].append([item_id, score])
                if attrs[item_id][1] == 1:
                    data2[user_id].append([item_id, score])
    return data1, data2

# 读取测试数据
def get_test_data(test_path):
    data = defaultdict(list)
    with open(test_path, 'r') as f:
        while (line := f.readline()) != '':
            user_id, num = map(int, line.strip().split('|'))
            for _ in range(num):
                line = f.readline()
                item_id = int(line.strip())
                data[user_id].append(item_id)
    return data

# 拆分数据为训练集和验证集
def split_data(data_user, ratio=0.85, shuffle=True):
    train_data, valid_data = defaultdict(list), defaultdict(list)
    for user_id, items in data_user.items():
        if shuffle:
            np.random.shuffle(items)
        split_point = int(len(items) * ratio)
        train_data[user_id] = items[:split_point]
        valid_data[user_id] = items[split_point:]
    return train_data, valid_data




