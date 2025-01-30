#!/bin/bash

echo 'How many members in ensemble?'
read number
for ((i = 1; i <= number; i++)) 
do
    python train_peppr.py $i
done

echo $number > ensemble_size.txt
