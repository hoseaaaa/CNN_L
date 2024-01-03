import os
import numpy as np
import sys

def clear_directory(directory):
    # Clear all txt files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)

def generate_and_save_matrices(directory, num, m, data_range=(0, 1)):
    for i in range(1, num + 1):
        filename = os.path.join(directory, f"{i}.txt")
        
        # Generate random matrix within the specified range
        matrix = np.random.uniform(data_range[0], data_range[1], size=(1, m))

        # Save matrix to file
        with open(filename, "w") as file:
            file.write(f"1 {m}\n")  # Write dimensions
            file.write("\n".join(map(str, matrix.flatten())) + "\n")  # Write matrix elements



# Generate and save matrices with specified data range
dataset_dir=str(sys.argv[1])
m =int(sys.argv[2])
num =  int(sys.argv[3])

# Clear existing files
clear_directory(dataset_dir)
data_min = 100  # Specify the minimum value for random data
data_max = 1000  # Specify the maximum value for random data
generate_and_save_matrices(dataset_dir, num, m, data_range=(data_min, data_max))
print(f"{num} matrices saved successfully in the {dataset_dir} directory. m: {m}  num  {num}")
