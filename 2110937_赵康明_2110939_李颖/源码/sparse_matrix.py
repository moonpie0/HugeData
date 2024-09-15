import numpy as np
import os

class Graph:
    def __init__(self):
        self.graph = {}
        self.all_nodes = set()

    def add_edge(self, from_node, to_node):
        """
        向图中添加一条边。
        """
        self.all_nodes.add(from_node)
        self.all_nodes.add(to_node)
        if from_node not in self.graph:
            self.graph[from_node] = []
        self.graph[from_node].append(to_node)

def load_data(file_path):
    """
    读取数据，构建图并返回图对象。
    """
    graph = Graph()

    with open(file_path, 'r') as file:
        for line in file:
            from_node, to_node = map(int, line.strip().split())
            graph.add_edge(from_node, to_node)

    return graph

def calculate_page_rank(graph, teleport_parameter=0.85, convergence_threshold=1e-12):

    all_nodes = graph.all_nodes
    num_nodes = len(all_nodes)
    initial_rank_new = (1 - teleport_parameter) / num_nodes
    difference = 1.0

    # 初始化节点的 PageRank 值为初始值
    old_rank = {node: initial_rank_new for node in all_nodes}

    # 迭代更新 PageRank 值，直到收敛
    iteration = 1
    print("开始迭代")
    while difference > convergence_threshold:
        new_rank = {node: initial_rank_new for node in all_nodes}

        # 遍历图中的每个节点，更新 PageRank 值
        for node, neighbors in graph.graph.items():
            # 计算节点的出链数
            num_outgoing = len(neighbors)
            # 更新节点的 PageRank 值
            for neighbor in neighbors:
                new_rank[neighbor] += teleport_parameter * old_rank[node] / num_outgoing

  
        s = sum(new_rank.values())
        adjustment = (1 - s) / num_nodes
        new_rank = {k: new_rank[k] + adjustment for k in new_rank}

        # 计算新旧 PageRank 值之差的绝对值之和
        differences = [abs(new_rank[node] - old_rank[node]) for node in all_nodes]
        difference = np.sum(differences)

        old_rank = new_rank

        print('迭代次数:', iteration, '差值:', difference)
        iteration += 1

    print('计算完成.')
    return old_rank


def get_top_nodes(page_ranks):
    """
    返回具有最高PageRank值的前100个节点及其PageRank值。
    """
    sorted_nodes = sorted(page_ranks.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes

def write_result(file_path, top_nodes):
    """
    将结果写入文件，确保数字不使用科学计数法。
    # """
    # if not os.path.exists(file_path):
    #     os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'w') as file:
        for node, rank in top_nodes:
            # 将数字转换为字符串，并确保不使用科学计数法
            file.write(f'{node} {format(rank, ".15f")}\n')

def main():
    if not os.path.exists('results/'):
        os.makedirs(os.path.dirname('results/'))
    graph = load_data('Data.txt')
    page_ranks = calculate_page_rank(graph)
    top_nodes = get_top_nodes(page_ranks)
    write_result('results/sparse_result.txt', top_nodes)
  


if __name__ == '__main__':
    main()
