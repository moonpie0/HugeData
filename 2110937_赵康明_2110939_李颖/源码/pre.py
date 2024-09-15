# 读取Data.txt文件中的数据
with open('Data.txt', 'r') as file:
    data = file.readlines()

# 统计行数
total_lines = len(data)

# 去除相同行
unique_lines = set(data)
total_unique_lines = len(unique_lines)

# 统计出现的数字（去除重复数字）
numbers = set()
srcNode=set()
destNode=set()
for line in data:
    parts = line.strip().split()
    src = int(line.split()[0].split()[0], 10)
    des = int(line.split()[1].split()[0], 10)
    numbers.update(parts)
    srcNode.add(src)
    destNode.add(des)

total_unique_numbers = len(numbers)
src_numbers=len(srcNode)
dest_numbers=len(destNode)
max_number = max(map(int, numbers))

print(f"links：{total_lines}")
print(f"unique links：{total_unique_lines}")
print(f"unique numbers:{total_unique_numbers}")
print(f"max number：{max_number}")
print(f"srcNode:{src_numbers}")
print(f"destNode:{dest_numbers}")
