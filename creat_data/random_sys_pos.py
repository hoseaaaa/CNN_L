import os
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import sys

def generate_symmetric_positive_definite_matrix(size):
    A = np.random.rand(size, size)
    A = 0.5 * (A + A.T)
    A += size * np.eye(size)
    return A

def clear_directory(directory):
    # Clear all txt files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)

def generate_sparse_matrices(filename_coo, filename_csr, size, density):
    dense_matrix = generate_symmetric_positive_definite_matrix(size)
    dense_matrix[dense_matrix > density] = 0

    # Ensure no zero diagonal elements
    np.fill_diagonal(dense_matrix, np.abs(np.random.rand(size)))

    coo_matrix_ = coo_matrix(dense_matrix)
    csr_matrix_ = csr_matrix(dense_matrix)

    # COO format
    with open(filename_coo, "w") as file:
        file.write(f"{coo_matrix_.shape[0]} {coo_matrix_.shape[1]} {len(coo_matrix_.data)}\n")
        file.write(" ".join(map(str, coo_matrix_.row)) + "\n")
        file.write(" ".join(map(str, coo_matrix_.col)) + "\n")
        file.write(" ".join(map(str, coo_matrix_.data)) + "\n")

    # CSR format
    with open(filename_csr, "w") as file:
        file.write(f"{csr_matrix_.shape[0]} {csr_matrix_.shape[1]} {len(csr_matrix_.data)}\n")
        file.write(" ".join(map(str, csr_matrix_.indptr)) + "\n")
        file.write(" ".join(map(str, csr_matrix_.indices)) + "\n")
        file.write(" ".join(map(str, csr_matrix_.data)) + "\n")


coo_dir=str(sys.argv[1])
csr_dir=str(sys.argv[4])

m = int(sys.argv[2])
num = int(sys.argv[3])
# Clear existing files

clear_directory(coo_dir)
clear_directory(csr_dir)

# Generate new files
for i in range(1, num+1):
    coo_filename = f"{coo_dir}/{i}.txt"
    csr_filename = f"{csr_dir}/{i}.txt"
    size = np.random.randint(m, m+1)
    density = np.random.uniform(0.1, 0.2)
    generate_sparse_matrices(coo_filename, csr_filename, size, density)
print(f"Files saved successfully: {coo_dir}, {csr_dir}  m: {m}   num : {num}" )
