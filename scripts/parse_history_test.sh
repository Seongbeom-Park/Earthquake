#!/bin/bash

prev_step=0
arrival_loss_sum=0
center_loss_sum=0
mag_loss_sum=0
sample_count=0
ADO_loss=0
RPV_loss=0
RSS_loss=0
USC_loss=0
depth_loss=0
lat_loss=0
long_loss=0
mag_loss=0
while read -r line ; do
	IFS_back=$IFS
	IFS=','
	data=($line)
	IFS=$IFS_back

	step=${data[0]}
	arrival_loss=${data[1]}
	center_loss=${data[2]}
	mag_loss=${data[3]}
	predict_ADO=${data[4]}
	predict_RPV=${data[5]}
	predict_RSS=${data[6]}
	predict_USC=${data[7]}
	predict_depth=${data[8]}
	predict_lat=${data[9]}
	predict_long=${data[10]}
	predict_mag=${data[11]}
	ADO=${data[12]}
	RPV=${data[13]}
	RSS=${data[14]}
	USC=${data[15]}
	depth=${data[16]}
	lat=${data[17]}
	long=${data[18]}
	mag=${data[19]}

	if [ "$prev_step" != "$step" ]; then
		echo "$prev_step,$arrival_loss_avg,$center_loss_avg,$mag_loss_avg,$ADO_avg,$RPV_avg,$RSS_avg,$USC_avg,$depth_avg,$lat_avg,$long_avg"
		prev_step=$step
		arrival_loss_sum=0
		center_loss_sum=0
		mag_loss_sum=0
		sample_count=0
		ADO_loss=0
		RPV_loss=0
		RSS_loss=0
		USC_loss=0
		depth_loss=0
		lat_loss=0
		long_loss=0
		mag_loss=0
	fi

	arrival_loss_sum=`bc <<< "scale=4; $arrival_loss_sum + $arrival_loss"`
	center_loss_sum=`bc <<< "scale=4; $center_loss_sum + center_loss"`
	mag_loss_sum=`bc <<< "scale=4; $mag_loss_sum + mag_loss"`
	sample_count=$(($sample_count + 1))
	ADO_loss=`bc <<< "scale=4; $ADO_loss + $ADO - $predict_ADO"`
	RPV_loss=`bc <<< "scale=4; $RPV_loss + $RPV - $predict_RPV"`
	RSS_loss=`bc <<< "scale=4; $RSS_loss + $RSS - $predict_RSS"`
	USC_loss=`bc <<< "scale=4; $USC_loss + $USC - $predict_USC"`
	depth_loss=`bc <<< "scale=4; $depth_loss + $depth - $predict_depth"`
	lat_loss=`bc <<< "scale=4; $lat_loss + "`
	long_loss=`bc <<< "scale=4; $long_loss + "`
	mag_loss=`bc <<< "scale=4; "`
done < "$1"
