import random

# 生成100个1到10之间的随机整数
random_numbers = [random.randint(1, 10) for _ in range(100)]

# 将生成的随机数写入 target.txt 文件
with open('./coo_dataset/target.txt', 'w') as file:
    for number in random_numbers:
        file.write(str(number) + ' ')
