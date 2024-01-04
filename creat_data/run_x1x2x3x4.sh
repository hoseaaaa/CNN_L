#!/bin/bash
set -e
current_dir=$(pwd)

if [ "$1" == "train" ]; then
    folder="train_dataset"
    echo "creat train x1x2x3x4  dataset "
elif [ "$1" == "test" ]; then
    folder="test_dataset"
    echo "creat test x1x2x3x4 dataset "
elif [ "$1" == "val" ]; then
    folder="val_dataset"
    echo "creat val x1x2x3x4 dataset "
else
    echo "Invalid argument. Please use 'train' or 'test' or 'val' "
    exit 1
fi

#lnn & lnnz
input_data_dir="../build/csr_dataset/"
input_data_dir="$input_data_dir$folder"

output_dir="../build/x_dataset/"
output_lnn="$output_dir$folder/lnn.txt"
output_lnnnz="$output_dir$folder/lnnnz.txt"

# lnn

> "$output_lnn"

# 遍历train文件夹中的所有数字结尾的文件夹
if [ -n "$2" ]; then Num=$2;  else  Num=40; fi

for i in $(seq 1 $Num); do
    input_file="$input_data_dir/${i}.txt"
    if [ -f "$input_file" ]; then

        lnn_number=$(awk 'NR==1{print $1}' "$input_file")
        lnn_value=$(echo "l($lnn_number)" | bc -l)
        lnnnz_number=$(awk 'NR==1{print $3}' "$input_file")
        lnnnz_value=$(echo "l($lnnnz_number)" | bc -l)

        # 输出结果到文件
        echo "$lnn_value" >> "$output_lnn"
        echo "$lnnnz_value" >> "$output_lnnnz"
    else
        echo "$input_file file not save "
    fi
done


# relax & eps_strong 

input_target_dir="../build/target/"
input_target_dir="$input_target_dir$folder"

output_relax="$output_dir$folder/relax.txt"
output_sita="$output_dir$folder/sita.txt"
>$output_relax
>$output_sita

for i in $(seq 1 $Num); do
    input_file="$input_target_dir/${i}.txt"
    if [ -f "$input_file" ]; then
        relax_value=$(grep -oP 'coarsening\.relax\s*:\s*\K[0-9.]+' "$input_file")
        sita_value=$(grep -oP 'aggr\.eps_strong\s*:\s*\K[0-9.]+' "$input_file")

        # 输出结果到文件
        echo "$relax_value" >> "$output_relax"
        echo "$sita_value" >> "$output_sita"
    else
        echo "$input_file file not save "
    fi
done

