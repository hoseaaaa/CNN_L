#!/bin/bash
set -e
# ../build/coo_dataset/    ../build/csr_dataset/
# train_dataset test_dataset val_dataset

COO_Dataset="../build/coo_dataset/" 
CSR_Dataset="../build/csr_dataset/"
B_Dataset="../build/b_dataset/"

if [ "$1" == "train" ]; then
    folder="train_dataset"
    echo "creat train dataset "
elif [ "$1" == "test" ]; then
    folder="test_dataset"
    echo "creat test dataset "
elif [ "$1" == "val" ]; then
    folder="val_dataset"
    echo "creat val dataset "
else
    echo "Invalid argument. Please use 'train' or 'test' or 'val' "
    exit 1
fi

if [ -n "$2" ]; then N=$2;  else  N=28; fi
if [ -n "$3" ]; then  Num=$3; else Num=100; fi

COO_target="$COO_Dataset$folder"
CSR_target="$CSR_Dataset$folder"
B_target="$B_Dataset$folder"

find $COO_target -type f -name "*.txt" -exec rm {} \;
find $CSR_target -type f -name "*.txt" -exec rm {} \;
find $B_target -type f -name "*.txt" -exec rm {} \;

python3 random_sys_pos.py $COO_target $N $Num $CSR_target
python3 random_b.py $B_target $N $Num
echo "successfully run dataset "
