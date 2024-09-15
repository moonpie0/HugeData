import networkx as nx

# 创建一个有向图
def pagerank(file_path):
    G = nx.DiGraph()
    with open(file_path, 'r') as file:
        for line in file:
            source_node, target_node = line.strip().split()
            G.add_edge(source_node, target_node)
    return nx.pagerank(G)




def write_results(pr, output_file):
    top_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    with open(output_file, 'w') as f:
        for node, rank in top_nodes:
            f.write(f"{node} {rank}\n")

def main():
    file_path = "Data.txt"
    output_file = "./results/basic_standard.txt"
    
    # 计算PageRank
    pr = pagerank(file_path)

    # 写入结果到文件
    write_results(pr, output_file)

if __name__ == "__main__":
    main()
