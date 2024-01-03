#!/bin/bash
set -e

if [ "$1" == "train" ]; then
    folder="train_dataset"
    echo "creat train target  dataset "
elif [ "$1" == "test" ]; then
    folder="test_dataset"
    echo "creat test target dataset "
elif [ "$1" == "val" ]; then
    folder="val_dataset"
    echo "creat val target dataset "
else
    echo "Invalid argument. Please use 'train' or 'test' or 'val' "
    exit 1
fi


input_dir="../build/csr_dataset/"
output_dir="../build/b_dataset/"

input_dir="$input_dir$folder"
output_dir="$output_dir$folder"
# 清空或创建 target.txt 文件
target_dir="../build/target/$folder/"
target_file="../build/target/$folder/target.txt"
find $target_dir -type f -name "*.txt" -exec rm {} \;
> $target_file

if [ -n "$2" ]; then Num=$2;  else  Num=100; fi

for i in $(seq 1 $Num); do
  input_file="$input_dir/${i}.txt"
  output_file="$output_dir/${i}.txt"
  output_info="../build/target/$folder/${i}.txt"

  # # 运行 dc_keti 并将输出保存到文件
  cp ../../AMGCL_KT/build/tutorial/6.power_grid_dc/dc_keti ./
  ./dc_keti "$input_file" "$output_file" > "$output_info"
  grep -oPi '\[ax\s*=\s*b\s*:\s*\K\d*\.?\d+' "$output_info" >> $target_file
  echo $output_info
  cd "$current_dir"
done
echo "successfully random target  "
