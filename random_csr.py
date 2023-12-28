import numpy as np
from scipy.sparse import csr_matrix

def generate_csr_sparse_matrix(filename, size, density):
    # 生成一个随机的稀疏矩阵
    dense_matrix = np.random.rand(size, size)
    dense_matrix[dense_matrix > density] = 0  # 根据稀疏度将一部分元素置为零
    sparse_matrix = csr_matrix(dense_matrix)

    # 获取CSR格式的行、列和数据
    rows, cols = sparse_matrix.nonzero()
    values = sparse_matrix.data

    # 保存到文件
    with open(filename, "w") as file:
        # 第一行: 行 列 非零元素个数
        file.write(f"{sparse_matrix.shape[0]} {sparse_matrix.shape[1]} {len(values)}\n")
        # 第二行: Index Pointers
        file.write(" ".join(map(str, sparse_matrix.indptr)) + "\n")
        # 第三行: Indices
        file.write(" ".join(map(str, sparse_matrix.indices)) + "\n")
        # 第四行: 数据
        file.write(" ".join(map(str, values)) + "\n")

# 生成1到10个稀疏矩阵文件
for i in range(1, ):
    filename = f"./dataset/{i}.txt"
    size = np.random.randint(28, 29)  # 随机生成矩阵的大小
    density = np.random.uniform(0.1, 0.5)  # 随机生成稀疏度
    generate_csr_sparse_matrix(filename, size, density)
    print(f"文件保存成功: {filename}")