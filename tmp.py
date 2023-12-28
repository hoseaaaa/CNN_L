import math

n = 27  # 设置重复次数

data = 66341492  # 将数字转换为字符串
# 4974178
# 8988782
# 11777726
result_base_e = math.log(data)
s_data = str(result_base_e)
# 重复字符串并在每次之间添加逗号
result_string = ','.join([s_data] * n)

print(result_string)
