#!/bin/bash
# $1: creat or not creat dataset 
# $2: train test val 
# $3: m
# $4: num matrix  
# creat data 
if [ "$1" == "creat" ]; then
    echo "creat  dataset "
    cd creat_data;
    ./run_dataset.sh $2 $3 $4  ;
    cd ../src
else
    echo "not creat dataset "
    cd src
fi

#train 
python3 train.py 
pwd

#test

#val 