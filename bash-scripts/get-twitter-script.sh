#!/usr/bin/bash

echo Starting!

declare -a numbers=("01" "02" "03" "09" "10")
cd data

for i in "${numbers[@]}"
do
	echo Getting "$i"
	wget https://archive.org/download/archiveteam-twitter-stream-2017-10/twitter_stream_2019_01_"$i".tar
	echo Got "$i" tar, un-tar now
	tar -xvf twitter_stream_2019_01_$i.tar
	echo Done un-tarring, writing now
	cd ~/6.s080
	python3 ec2-data.py $i
	cd data
	rm -r $i
	rm twitter_stream_2019_01_"$i".tar
	echo Done with "$i"!
done