import os
import numpy as np

def clear_directory(directory):
    # Clear all txt files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)

def generate_and_save_matrices(directory, m, n):
    for i in range(1, m + 1):
        filename = os.path.join(directory, f"{i}.txt")
        matrix = np.random.rand(1, n)

        # Save matrix to file
        with open(filename, "w") as file:
            file.write(f"1 {n}\n")  # Write dimensions
            file.write("\n".join(map(str, matrix.flatten())) + "\n")  # Write matrix elements

# Clear existing files
clear_directory("./b_dataset")

# Generate and save matrices
m = 100  # You can change this to the desired number of matrices
n = 28   # You can change this to the desired number of columns
generate_and_save_matrices("./b_dataset", m, n)
print(f"{m} matrices saved successfully in the b_dataset directory.")
