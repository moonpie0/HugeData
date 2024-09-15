import os

file1_path = './results/basic_standard.txt'
file2_path = './results/Block_Strip_result.txt'

# 提取文件名
file1_name = os.path.basename(file1_path)
file2_name = os.path.basename(file2_path)

# 读取第一个文本文件的第一列并转换为列表
with open(file1_path, 'r') as file1:
    file1_lines = file1.readlines()[:100]
    file1_column1 = [line.split()[0] for line in file1_lines]

# 读取第二个文本文件的第一列并转换为列表
with open(file2_path, 'r') as file2:
    file2_lines = file2.readlines()[:100]
    file2_column1 = [line.split()[0] for line in file2_lines]

# 找出不同位置的索引
different_indices = []
count = 0
for i in range(min(len(file1_column1), len(file2_column1))):
    if file1_column1[i] != file2_column1[i]:
        count += 1
        different_indices.append(i)

print(f"文件 '{file1_name}' 和 '{file2_name}' 的按顺序不相同的元素个数：", count)
print("不相同的索引位置：", different_indices)
