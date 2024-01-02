import os
import numpy as np

def clear_directory(directory):
    # Clear all txt files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)

def generate_and_save_matrices(directory, m, n, data_range=(0, 1)):
    for i in range(1, m + 1):
        filename = os.path.join(directory, f"{i}.txt")
        
        # Generate random matrix within the specified range
        matrix = np.random.uniform(data_range[0], data_range[1], size=(1, n))

        # Save matrix to file
        with open(filename, "w") as file:
            file.write(f"1 {n}\n")  # Write dimensions
            file.write("\n".join(map(str, matrix.flatten())) + "\n")  # Write matrix elements

# Clear existing files
clear_directory("./b_dataset")

# Generate and save matrices with specified data range
m = 20  # You can change this to the desired number of matrices
n = 2000   # You can change this to the desired number of columns
data_min = 100  # Specify the minimum value for random data
data_max = 1000  # Specify the maximum value for random data
generate_and_save_matrices("./b_dataset", m, n, data_range=(data_min, data_max))
print(f"{m} matrices saved successfully in the b_dataset directory.")
