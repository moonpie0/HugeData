import math
import os
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
import random
from datetime import datetime
from math import sqrt
import numpy as np

if not os.path.exists('tmp/'):
    os.makedirs(os.path.dirname('tmp/'))
if not os.path.exists('result/'):
    os.makedirs(os.path.dirname('result/'))

TRAIN_FILE = 'data/train.txt'
TEST_FILE = 'data/test.txt'

TRAIN_FILE2 = 'tmp/train_data.txt'
TEST_FILE2 = 'tmp/test_data.txt'
VALIDATION_FILE = 'tmp/validation_data.txt'

SPARSE_MATRIX_FILE = 'tmp/sparse_matrix1.txt'
USER_AVERANGE_FILE = 'tmp/user_average1.txt'
ITEM_LIST_FILE = 'tmp/item_list1.txt'

RESULT_FILE = 'result/result1.txt'

user_num = 19835
item_num = 624961

#相似度矩阵
#correlation_matrix = []
matrix_dict = {}
#物品列表
item_list = set()
#用户评分矩阵
user_matrix = []
#用户平均评分
user_averange = {}

#挑选相似的用户数量
N = 500


def split_data(in_file=TRAIN_FILE, ratio=0.01, data_random=True):
    if os.path.exists('tmp/train_data.txt'):
        os.remove('tmp/train_data.txt')
    if os.path.exists('tmp/validation_data.txt'):
        os.remove('tmp/validation_data.txt')
    if os.path.exists('tmp/test_data.txt'):
        os.remove('tmp/test_data.txt')


    with open(in_file, 'r') as file:
        line = file.readline().strip()
        while line:
            user_id, num_items = line.split('|')
            user_id = int(user_id)
            num_items = int(num_items)
            items = []
            for _ in range(num_items):
                item_id, score = file.readline().strip().split()
                items.append((item_id, score))

            #打乱顺序
            if data_random:
                random.shuffle(items)

            #分割
            index = int(len(items) * ratio)
            train_items = items[index:]
            validation_items = items[:index]

            with open('tmp/train_data.txt', 'a') as train_file:
                #if len(train_items) > 0:
                train_file.write(f"{user_id}|{len(train_items)}\n")
                for item in train_items:
                    train_file.write(f"{item[0]}  {item[1]}\n")

            with open('tmp/validation_data.txt', 'a') as validation_file:
                validation_file.write(f"{user_id}|{len(validation_items)}\n")
                for item in validation_items:
                    validation_file.write(f"{item[0]}  {item[1]}\n")

            with open('tmp/test_data.txt', 'a') as validation_file:
                validation_file.write(f"{user_id}|{len(validation_items)}\n")
                for item in validation_items:
                    validation_file.write(f"{item[0]}\n")

            line = file.readline().strip()


def sparse_matrix(in_file=TRAIN_FILE, out_file1=SPARSE_MATRIX_FILE, out_file2=USER_AVERANGE_FILE, out_file3 = ITEM_LIST_FILE):
    '''
    将训练集转换为稀疏矩阵形式，并计算每个用户的平均评分
    :param in_file:
    <user id>|<numbers of rating items>
    <item id>   <score>
    :param out_file1: <user id> <item id> <score>
    :param out_file2: <user id> <average score>
    :return:
    '''
    count = 1
    item_list = set()

    with open(in_file, 'r') as file:
        line = file.readline().strip()  # 读取第一行并去除首尾空格
        while line:
            user_id, num_items = line.split('|')
            score_sum = 0
            num_items = int(num_items)
            # 读取 num_items 行数据并进行相关操作
            for _ in range(num_items):
                item_id, score = file.readline().strip().split()
                score_sum += int(score)
                if count%10000 == 0:
                    print("进度", (count/5001507)*100, "%")
                count += 1
                with open(out_file1, 'a') as file2:
                    file2.write(f"{user_id} {item_id} {score}\n")
                item_list.add(item_id)

            with open(out_file2, 'a') as file3:
                aver = float(score_sum / num_items)
                file3.write(f"{user_id} {aver}\n")

            line = file.readline().strip()

    with open(out_file3, 'w') as file:
        # 将集合中的元素逐行写入文件
        for element in item_list:
            file.write(str(element) + '\n')
    print("Convert to a sparse matrix!\n")


def calculate_similarity( sparse_file = SPARSE_MATRIX_FILE):
    '''
    计算用户之间的相似度(使用库计算)
    :return: correlation matrix 相似度矩阵
    '''

    user_ids = []
    item_ids = []
    scores = []

    with open(sparse_file, 'r') as file:
        for line in file:
            user_id, item_id, score = line.strip().split()
            user_ids.append(int(user_id))
            item_ids.append(int(item_id))
            scores.append(float(score))


    matrix = coo_matrix((scores, (user_ids, item_ids)), shape=(user_num, item_num+1))

    correlation_matrix = cosine_similarity(matrix)
    print(correlation_matrix.shape)
    #print(correlation_matrix)

    #这个文件有6G，不写进文件了，直接存在内存里
    return correlation_matrix


def calculate_similarity_2():
    '''
    计算用户之间的相似度（手工计算）
    :return: correlation matrix 相似度矩阵
    '''

    correlation_matrix = np.zeros((user_num, user_num))
    count = 0

    #遍历user_num次
    for i in range(user_num):
        i_item_count = len(user_matrix[i])
        #用户i对其他物品的评分与均值之差
        tmp1 = np.sum([math.pow(user_matrix[i][item] - user_averange[i], 2) for item in user_matrix[i]])
        for j in range(i+1, user_num):
            j_item_count = len(user_matrix[j])
            x = 0 #分子
            # 用户j对其他物品的评分与均值之差
            tmp2 = np.sum([math.pow(user_matrix[j][item] - user_averange[j], 2) for item in user_matrix[j]])

            #从更少评分物品的列表开始遍历，会更节省时间
            if i_item_count<= j_item_count:
                for item in user_matrix[i]:
                    #有共同评分的物品
                    if user_matrix[j].get(item) is not None:
                        x1 = user_matrix[i][item] - user_averange[i]
                        x2 = user_matrix[j][item] - user_averange[j]
                        x += x1 * x2
                    else:
                        continue
            else:
                for item in user_matrix[j]:
                    # 有共同评分的物品
                    if user_matrix[i].get(item) is not None:
                        x1 = user_matrix[i][item] - user_averange[i]
                        x2 = user_matrix[j][item] - user_averange[j]
                        x += x1 * x2
                    else:
                        continue

            if tmp1== 0 or tmp2 == 0:
                correlation_matrix[i][j] = 0
                correlation_matrix[j][i] = 0
            else:
                correlation_matrix[i][j] = x / (math.sqrt(tmp1 * tmp2))
                correlation_matrix[j][i] = correlation_matrix[i][j]
        count +=1
        if count % 100 == 0:
            print("已经计算了",count, "轮")


    return correlation_matrix




def load_data():
    # 导入item_list
    #item_list = set()
    with open(ITEM_LIST_FILE, 'r') as file:
        for line in file:
            item_list.add(int(line.strip()))

    # 导入user_averange
    #user_averange = []
    with open(USER_AVERANGE_FILE, 'r') as file:
        for line in file:
            # 分割每行数据，以空格分隔
            data = line.strip().split()
            # 提取id和score
            id = int(data[0])
            score = float(data[1])
            # 将id和score存储到字典中
            user_averange[id] = score

    #导入用户评分矩阵
    with open(TRAIN_FILE, 'r') as file:
        for line in file:
            user_id, num_items = line.split('|')
            user_id = int(user_id)
            while len(user_matrix) < user_id+1 :
                user_matrix.append({})
            num_items = int(num_items)
            for _ in range(num_items):
                item_id, score = file.readline().strip().split()
                item_id = int(item_id)
                score = int(score)
                user_matrix[user_id][item_id] = score
        line = file.readline().strip()

    #print(user_matrix[0])
    #if 518385 in user_matrix[0]:
    #    print("yes")
    print("Data import completed, starting test.")

def predict(user, item, correlation_matrix):
    #print("This is predict function")
    x = 0
    y = 0 #相似度之和
    count = 0
    #遍历每一位用户
    for u in range (user_num):
        if item in user_matrix[int(u)] and correlation_matrix[user][int(u)] >= 0.1:
            #print(correlation_matrix[user][int(u)])
            x += correlation_matrix[user][int(u)] * (user_matrix[int(u)][item] - user_averange[int(u)])
            y += correlation_matrix[user][int(u)]
        #if count == 500
        #    break

    if y == 0:
        return user_averange[user]
    else:

        return (x/y + user_averange[user])








def test(correlation_matrix, test_path=TEST_FILE, result_file = RESULT_FILE):
    '''
    测试
    :param correlation_matrix: 相似度矩阵
    '''
    predict_matrix = []
    count = 0

    with open(test_path, 'r') as file:
        line = file.readline().strip()  # 读取第一行并去除首尾空格
        while line:
            if count%1000==0:
                print("test:", count)
            user_id, num_items = line.split('|')
            user_id = int(user_id)
            num_items = int(num_items)
            #拓展predic_matrix，防止数组越界
            while len(predict_matrix)< user_id + 1 :
                predict_matrix.append([])
            count += 1
            # 读取 num_items 行数据并进行相关操作
            for _ in range(num_items):
                item_id = int(file.readline().strip())
                if item_id not in item_list:
                    p = user_averange[user_id]
                else :
                    p = predict(user_id, item_id, correlation_matrix)
                predict_matrix[user_id].append((item_id, p))


            line = file.readline().strip()



    with open(result_file, 'w') as file:
        for i in range (len(predict_matrix)):
            file.write(str(i) + "|" + str(len(predict_matrix[i]))+ "\n")
            for j in range(len(predict_matrix[i])):
                score = max(0, min(100, round(predict_matrix[i][j][1])))
                file.write(str(predict_matrix[i][j][0]) + " " + str(score) + "\n")

    print("Finish test!")




def evaluate(val_file=VALIDATION_FILE, res_file=RESULT_FILE):
    """
        比较validate_data.txt的结果与result.txt中的结果，计算RMSE
    """
    with open(val_file, "r") as val, open(res_file, "r") as res:
        RMSE = 0
        count = 0

        for line_val, line_res in zip(val, res):
            user_id, num_items = line_val.strip().split('|')
            user_id_r, num_items_r = line_res.strip().split('|')
            assert int(user_id) == int(user_id_r) and int(num_items) == int(num_items_r)
            count += int(num_items)

            for _ in range(int(num_items)):
                item_id, score = val.readline().strip().split()
                item_id_r, score_r = res.readline().strip().split()
                print(f"score_id: {score} score_r:{score_r}")
                assert item_id == item_id_r
                RMSE += (float(score) - float(score_r)) ** 2

        RMSE = math.sqrt(RMSE / count)
        print(f"最终测试结果: test数量：{count}, RMSE = {RMSE}")
        return RMSE


def main():
    if os.path.exists(SPARSE_MATRIX_FILE):
        os.remove(SPARSE_MATRIX_FILE)
    if os.path.exists(USER_AVERANGE_FILE):
        os.remove(USER_AVERANGE_FILE)


sparse_matrix_not_exists = True


if __name__ == '__main__':
    #main()
    '''
    训练验证集，可以计算RMSE
    '''
    split_data()
    if sparse_matrix_not_exists:  #已有sparse_matrix.txt则不需要
        sparse_matrix(TRAIN_FILE2)
    current_time = datetime.now()
    load_data()
    correlation_matrix = calculate_similarity()
    #correlation_matrix = calculate_similarity_2()
    test(correlation_matrix, TEST_FILE2)
    current_time2 = datetime.now()
    evaluate()

    print(f"验证集训练总耗时：{current_time2 - current_time}")

    main()

    '''
    训练train.txt
    '''
    if sparse_matrix_not_exists:  #已有sparse_matrix.txt则不需要
        sparse_matrix(TRAIN_FILE)
    current_time3 = datetime.now()
    load_data()
    correlation_matrix = calculate_similarity()
    #correlation_matrix = calculate_similarity_2()
    test(correlation_matrix, TEST_FILE)
    current_time4 = datetime.now()

    print(f"验证集训练总耗时：{current_time4 - current_time3}")



