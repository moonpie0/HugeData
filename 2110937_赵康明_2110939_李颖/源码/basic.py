import numpy as np
import os

class Graph:
    def __init__(self):
        self.graph = {}
        self.all_nodes = set()

    def add_edge(self, from_node, to_node):
        """
        添加一条边到图中。
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

def matrix_multiply(A, B):
    """
    计算矩阵 A 和向量 B 的乘积。
    """
    # 获取矩阵 A 的行数和列数
    m, n = A.shape
    # 获取向量 B 的长度
    p = len(B)
    # 确保矩阵 A 的列数和向量 B 的长度相等
    assert n == p, "The number of columns in A must be equal to the length of B."

    # 初始化结果向量
    result = np.zeros(m, dtype=np.float64)
    
    # 计算矩阵乘法
    for i in range(m):
        for j in range(n):
            result[i] += A[i, j] * B[j]

    return result




def calculate_page_rank(graph, teleport_parameter=0.85 ,tol=1e-15, max_iterations=1000):
    """
    基于给定的图、节点和参数,计算PageRank值。
    """
    all_nodes = graph.all_nodes
    N = len(all_nodes)
    node_idx = {node: i for i, node in enumerate(sorted(all_nodes))}

    # 构建转移矩阵M
    M = np.zeros([N, N], dtype=np.float64)
    
    # 构建转移矩阵S
    S = np.zeros([N, N], dtype=np.float64)
    for out_node, in_nodes in graph.graph.items():
        for in_node in in_nodes:
            S[node_idx[in_node], node_idx[out_node]] = 1

    # 处理矩阵
    for col in range(N):
        sum_of_col = S[:, col].sum()
        if sum_of_col == 0:
            S[:, col] = 1 / N
        else:
            S[:, col] /= sum_of_col

    # 计算总转移矩阵M
    E = np.ones((N, N), dtype=np.float64)
    M = teleport_parameter * S + (1 - teleport_parameter) * E / N


    # 初始化PageRank值
    P = np.ones(N, dtype=np.float64) / N

    # 迭代更新PageRank值，直到收敛或达到最大迭代次数
    for iteration in range(max_iterations):
        prev_P = np.copy(P)
        ## P=MP
        ## P=M@p
        P=np.dot(M,P)
        # P = matrix_multiply(M, P)
        diff = np.linalg.norm(P - prev_P)
        if diff < tol:
            print(f'Converged after {iteration + 1} iterations.')
            break

    if iteration == max_iterations - 1:
        print('Maximum number of iterations reached without convergence.')

    return P, node_idx

def top_nodes(pr, node_idx):
    """
    返回所有节点及其PageRank值，并按照PageRank值降序排列。
    """
    sorted_nodes = sorted(node_idx.items(), key=lambda x: pr[x[1]], reverse=True)
    return [(node, pr[index]) for node, index in sorted_nodes]

def write_result(file_path, sorted_nodes):
    """
    将排好序的所有节点及其PageRank值写入文件。
    """
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'w') as file:
        for node, rank in sorted_nodes:
            file.write(f'{node} {rank}\n')

def main():
    graph = load_data('Data.txt')
    pr, node_idx = calculate_page_rank(graph)
    sorted_nodes = top_nodes(pr, node_idx)
    write_result('./results/basic_result.txt', sorted_nodes)
    print("finish ")

if __name__ == '__main__':
    main()
