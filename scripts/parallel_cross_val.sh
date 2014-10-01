 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 
python eval.py --resultfile ../data/results/results_Sep30_r0.1n5_mul_s_c1.0.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/FINAL_R0.1N7_mul_s_c\=1.0_d\=1_bellman.h5 --parallel --search_parallel
 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 
python eval.py --resultfile ../data/results/results_Sep30_r0.1n5_mul_s_c10.0.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/FINAL_R0.1N7_mul_s_c\=10.0_d\=1_bellman.h5 --parallel --search_parallel
 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 
python eval.py --resultfile ../data/results/results_Sep30_r0.1n5_mul_s_c100.0.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/FINAL_R0.1N7_mul_s_c\=100.0_d\=1_bellman.h5 --parallel --search_parallel
 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 
python eval.py --resultfile ../data/results/results_Sep30_r0.1n5_mul_s_c1000.0.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/FINAL_R0.1N7_mul_s_c\=1000.0_d\=1_bellman.h5 --parallel --search_parallel

 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 
# mul_s features r0.1, n5, c 1.0->1000.0
python eval.py --resultfile ../data/results/results_Sep30_r0.1n7_mul_s_c1.0.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/FINAL_R0.1N7_mul_s_c\=1.0_d\=1_bellman.h5 --parallel --search_parallel
 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 
python eval.py --resultfile ../data/results/results_Sep30_r0.1n7_mul_s_c10.0.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/FINAL_R0.1N7_mul_s_c\=10.0_d\=1_bellman.h5 --parallel --search_parallel
 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 
python eval.py --resultfile ../data/results/results_Sep30_r0.1n7_mul_s_c100.0.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/FINAL_R0.1N7_mul_s_c\=100.0_d\=1_bellman.h5 --parallel --search_parallel
 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 
python eval.py --resultfile ../data/results/results_Sep30_r0.1n7_mul_s_c1000.0.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/FINAL_R0.1N7_mul_s_c\=1000.0_d\=1_bellman.h5 --parallel --search_parallel

 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 

### mul_grip FEATURES ###
# mul_grip features r0.1, n5, c 1.0->1000.0
python eval.py --resultfile ../data/results/results_Sep30_r0.1n5_mul_grip_c1.0.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type mul_grip --weightfile ../data/weights/FINAL_R0.1N7_mul_grip_c\=1.0_d\=1_bellman.h5 --parallel --search_parallel
 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 
python eval.py --resultfile ../data/results/results_Sep30_r0.1n5_mul_grip_c10.0.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type mul_grip --weightfile ../data/weights/FINAL_R0.1N7_mul_grip_c\=10.0_d\=1_bellman.h5 --parallel --search_parallel
 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 
python eval.py --resultfile ../data/results/results_Sep30_r0.1n5_mul_grip_c100.0.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type mul_grip --weightfile ../data/weights/FINAL_R0.1N7_mul_grip_c\=100.0_d\=1_bellman.h5 --parallel --search_parallel
 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 
python eval.py --resultfile ../data/results/results_Sep30_r0.1n5_mul_grip_c1000.0.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type mul_grip --weightfile ../data/weights/FINAL_R0.1N7_mul_grip_c\=1000.0_d\=1_bellman.h5 --parallel --search_parallel

 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 
python eval.py --resultfile ../data/results/results_Sep30_r0.1n7_mul_grip_c1.0.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type mul_grip --weightfile ../data/weights/FINAL_R0.1N7_mul_grip_c\=1.0_d\=1_bellman.h5 --parallel --search_parallel
 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 
python eval.py --resultfile ../data/results/results_Sep30_r0.1n7_mul_grip_c10.0.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type mul_grip --weightfile ../data/weights/FINAL_R0.1N7_mul_grip_c\=10.0_d\=1_bellman.h5 --parallel  --search_parallel
 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 
python eval.py --resultfile ../data/results/results_Sep30_r0.1n7_mul_grip_c100.0.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type mul_grip --weightfile ../data/weights/FINAL_R0.1N7_mul_grip_c\=100.0_d\=1_bellman.h5 --parallel --search_parallel
 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 
python eval.py --resultfile ../data/results/results_Sep30_r0.1n7_mul_grip_c1000.0.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type mul_grip --weightfile ../data/weights/FINAL_R0.1N7_mul_grip_c\=1000.0_d\=1_bellman.h5 --parallel --search_parallel


 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 
# Baseline on r0.1, r0.15, n5, n7

python eval.py --resultfile ../data/results/results_Sep30_r0.1n5_greedy.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 greedy finger bij --parallel --search_parallel

 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 
python eval.py --resultfile ../data/results/results_Sep30_r0.1n7_greedy.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 greedy finger bij --parallel --search_parallel

 
ipcluster stop --profile=ssh
sleep 30
 
ipcluster start --profile=ssh --daemonize
sleep 90
 
python eval.py --resultfile ../data/results/results_Sep30_r0.15n5_greedy.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 greedy finger bij --parallel --search_parallel

