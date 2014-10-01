#!/bin/bash
 
ipcluster stop --profile=ssh
sleep 30


for i in {20..90..10};
do
    ipcluster start --profile=ssh --daemonize
    sleep 90
    echo STARTING TASKS $i through $(($i+1))
    python eval.py --i_start $i --i_end $(($i+10)) --landmarkfile ../data/misc/landmarks_r0.15n7.h5 --resultfile ../data/results/reproducible_results_r0.1n7_landmark_w5d2.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type landmark --weightfile ../data/weights/Sep18_final_r0.15n7_landmark_c\=100.0_d\=500.0_bellman.h5 --width 5 --depth 2 --search_parallel --parallel
 
    ipcluster stop --profile=ssh
    sleep 30
   
done


