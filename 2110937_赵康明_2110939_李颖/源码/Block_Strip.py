import json
import os
import numpy as np
import math
import pickle
from collections import defaultdict, OrderedDict

block_size = 50
b = 0.85
E = 1e-15
DATA_STRIP = 'data_new/'
RESULT='results/'
TEMP_DATA = 'tmp.txt'

#按dest重新排序
def sort_by_dest(data_path, new_data_path):
    edges = []
    out_of_nodes = defaultdict(int)
    N=set()
    with open(data_path, 'r') as f:
        for line in f:
            src, dest = map(int, line.strip().split())
            out_of_nodes[src] += 1 #统计出度
            #统计节点个数
            N.add(src)
            N.add(dest)
            if (src, dest) not in edges:
                edges.append((src, dest))

    # 按dest排序
    sorted_edges = sorted(edges, key=lambda x: x[1])
    # 写入新的数据文件
    with open(new_data_path, 'w') as f:
        for edge in sorted_edges:
            f.write(str(edge[0]) + " " + str(edge[1]) + "\n")

    #N与out_of_nodes写入文件
    with open(TEMP_DATA, 'w') as f:
        f.write(str(len(N)) + "\n")
        for i in out_of_nodes:
            f.write(str(i) + " " + str(out_of_nodes[i]) + "\n")


def process_data(N, new_data_path, out_of_nodes, block_size) :
    # 计算需要划分的块数
    k = math.ceil(N / block_size)

    temp_dict = dict()
    num = 0
    with open(new_data_path, 'r') as f:
        for line in f:
            src, dest = map(int, line.strip().split())

            #dest超过范围则保存
            if(dest > (block_size + num * block_size)) :
                temp_dict= OrderedDict(sorted(temp_dict.items(), key=lambda x: int(x[0])))
                save_file_name = os.path.join(DATA_STRIP, f'{num}.txt')
                with open(save_file_name, 'w', encoding='utf-8') as out:
                    out.write(json.dumps(temp_dict))
                temp_dict = dict() #清空
                num +=1

            #存入临时字典
            if src not in temp_dict:
                temp_dict[src] = [out_of_nodes[src], [dest]]
            else:
                temp_dict[src][1].append(dest)

    #保存剩余数据
    if N % block_size != 0:
        temp_save_dict = OrderedDict(sorted(temp_dict.items(), key=lambda x: int(x[0])))
        save_file_name = os.path.join(DATA_STRIP, f'{num}.txt')
        with open(save_file_name, 'w', encoding='utf-8') as out:
            out.write(json.dumps(temp_save_dict))


    print("Finish Block Preprocessing")


def block_pagerank(size, N):
    # 初始化列向量 r_old
    r_old = {}
    for src in range(1, N+1):
        r_old[src] = np.float64(1/N)

    r_new = np.zeros(N+1)

    if_end = 0
    strip_list = os.listdir(DATA_STRIP) #文件列表
    K = N // size
    if N % size != 0:
        K += 1

    round = 1
    while not if_end:
        S = 0

        # 遍历每一文件（分割后的块）
        for strip_file in range(len(strip_list)):
            strip_path = DATA_STRIP + str(strip_file) + '.txt'  #分割后数据
            with open(strip_path, 'r', encoding='utf-8') as f1:
                strip = json.loads(f1.read())  #读取数据
            if strip == {}:
                continue
            else: #文件数据不为空
                for src in strip:
                    degree = strip[src][0]
                    r_old_i = r_old[int(src,10)]
                    for dest in strip[src][1]:
                        r_new[dest-0] += b * r_old_i / degree

        e = 0.0

        #处理Dead ends和spider trap
        for i in range(1, N+1):
            S += r_new[i]

        for dest in range(1, N+1):
            r_new[dest] += np.float64((1 - S) / N)


        #最大误差e
        for i in range(1, N+1):
            diff = abs(r_old[i] - r_new[i])
            r_old[i] = r_new[i] #更新r_old
            if e < diff :
                e = diff

        print("count:", round, " e:", e)
        round += 1

        if abs(e) <= E:
            if_end = 1
        else:
            r_new = np.zeros(N+1)

    print("Finish PageRank!\n")
    return r_old


def main():
    #若已有data_new分割文件和tmp.txt文件，可注释sort_by_dest函数和process_data函数
    sort_by_dest('Data.txt', 'new_data.txt')

    #从文件中获取N和out_of_nodes
    out_of_nodes = defaultdict(int)
    with open(TEMP_DATA, 'r') as f:
        N = int(f.readline().rstrip(),10)
        for line in f:
            dest, count = line.rstrip().split( )
            out_of_nodes[int(dest)] = int(count)


    process_data(N, 'new_data.txt', out_of_nodes, block_size)
    r_new = block_pagerank(block_size, N)

    result = dict()
    result = dict(sorted(r_new.items(), key=lambda d: d[1], reverse=True))  # 按照rank排序

    file_path = RESULT + 'Block_Strip_result.txt'

    with open(file_path, 'w', encoding='utf-8') as f:
        for i in result:
            f.write(str(i) + ' ' + str(result[i]) + '\n')

    # 取出前100个value最大的key
    top_100_keys = list(result.keys())[:100]

    # 打印出前100个value最大的<key, value>
    print("PageRankTop 100:")
    for key in top_100_keys:
        print(key, result[key])


if __name__ == '__main__':
    if not os.path.exists(DATA_STRIP):
        os.makedirs(os.path.dirname(DATA_STRIP))
    if not os.path.exists(RESULT):
        os.makedirs(os.path.dirname(RESULT))
    main()