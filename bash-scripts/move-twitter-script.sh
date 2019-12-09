#!/bin/bash

echo Starting!

declare -a numbers=("20" "21" "22" "23" "24" "25" "26" "27" "28" "29" "30" "31")

for i in "${numbers[@]}"
do
	scp -i aws-key.pem ec2-user@\[ec2-3-82-205-207.compute-1.amazonaws.com\]:~/6.s080/data/01"$i".csv /mnt/c/Users/steph/Documents/6.s080/6s080_final_project/twitter-data/01"$i".csv
done