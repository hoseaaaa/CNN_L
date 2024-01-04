#!/bin/bash
# $1: creat or not creat dataset 
# $2: train test val 
# $3: m
# $4: num matrix  
# creat data 
set -e

m=$3 #weidu
num=$4 #geshu
current_dir=$(pwd)

if [ "$1" == "creat" ]; then
    echo "creat  dataset "
    cd creat_data;
    ./run_dataset.sh $2 $m $num  ;
    ./random_target.sh $2 $num  ;
    ./run_x1x2x3x4.sh $2 $num
    cd "$current_dir"
    echo "suss creat $2 data weidu:$m  num : $num "
elif [ "$1" == "train" ]; then
    echo " train model " & cd src ;  
    python3 train.py
elif [ "$1" == "test" ]; then
    echo "test model " & cd src ;  

elif [ "$1" == "val" ]; then
    echo "val model " & cd src ;  
else
    echo "error ,please input creat | train | test | val "
fi