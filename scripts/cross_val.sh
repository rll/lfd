# Scripts to run cross validation on different feature types and values of C

echo RUNNING Baseline
# no features
#python eval.py --animation 0 --i_end 100 --resultfile ../data/results/results_9_10_no_feature.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Jul_3_0.1_test.h5 greedy finger bij
'''
echo RUNNING base features
# base features
python eval.py --animation 0 --i_end 100 --resultfile ../data/results/results_9_10_base_c\=1_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Jul_3_0.1_test.h5 feature finger bij --feature_type base --weightfile ../data/weights/labels_Jul_3_0.1_base_c\=1_bellman.h5

python eval.py --animation 0 --i_end 100 --resultfile ../data/results/results_9_10_base_c\=10.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Jul_3_0.1_test.h5 feature finger bij --feature_type base --weightfile ../data/weights/labels_Jul_3_0.1_base_c\=10.0_bellman.h5

python eval.py --animation 0 --i_end 100 --resultfile ../data/results/results_9_10_base_c\=100.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Jul_3_0.1_test.h5 feature finger bij --feature_type base --weightfile ../data/weights/labels_Jul_3_0.1_base_c\=100.0_bellman.h5

python eval.py --animation 0 --i_end 100 --resultfile ../data/results/results_9_10_base_c\=1000.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Jul_3_0.1_test.h5 feature finger bij --feature_type base --weightfile ../data/weights/labels_Jul_3_0.1_base_c\=1000.0_bellman.h5

echo RUNNING mul features
# mul features
python eval.py --animation 0 --i_end 100 --resultfile ../data/results/results_9_10_mul_c\=10.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Jul_3_0.1_test.h5 feature finger bij --feature_type mul --weightfile ../data/weights/labels_Jul_3_0.1_mul_c\=10.0_bellman.h5

python eval.py --animation 0 --i_end 100 --resultfile ../data/results/results_9_10_mul_c\=100.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Jul_3_0.1_test.h5 feature finger bij --feature_type mul --weightfile ../data/weights/labels_Jul_3_0.1_mul_c\=100.0_bellman.h5

python eval.py --animation 0 --i_end 100 --resultfile ../data/results/results_9_10_mul_c\=1000.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Jul_3_0.1_test.h5 feature finger bij --feature_type mul --weightfile ../data/weights/labels_Jul_3_0.1_mul_c\=1000.0_bellman.h5

# Baseline on r0.1, r0.15, n5, n7
python eval.py --resultfile ../data/results/results_Sep18_r0.1n5_greedy.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 greedy finger bij --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n7_greedy.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 greedy finger bij --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n5_greedy.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 greedy finger bij --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n7_greedy.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n7.h5 greedy finger bij --gpu



### mul FEATURES ###
# mul features r0.1, n5, c 1.0->1000.0
python eval.py --resultfile ../data/results/results_Sep18_r0.1n5_mul_c\=1.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type mul --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_c\=1.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n5_mul_c\=10.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type mul --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_c\=10.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n5_mul_c\=100.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type mul --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_c\=100.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n5_mul_c\=1000.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type mul --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_c\=1000.0_bellman.h5 --gpu

# mul features r0.15, n5, c 1.0->1000.0
python eval.py --resultfile ../data/results/results_Sep18_r0.15n5_mul_c\=1.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type mul --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_c\=1.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.15n5_mul_c\=10.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type mul --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_c\=10.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.15n5_mul_c\=100.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type mul --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_c\=100.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.15n5_mul_c\=1000.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type mul --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_c\=1000.0_bellman.h5 --gpu

# mul features r0.1, n7, c 1.0->1000.0
python eval.py --resultfile ../data/results/results_Sep18_r0.1n7_mul_c\=1.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type mul --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_c\=1.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n7_mul_c\=10.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type mul --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_c\=10.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n7_mul_c\=100.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type mul --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_c\=100.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n7_mul_c\=1000.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type mul --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_c\=1000.0_bellman.h5 --gpu


### mul_s FEATURES ###
# mul_s features r0.1, n5, c 1.0->1000.0
python eval.py --resultfile ../data/results/results_Sep18_r0.1n5_mul_s_c\=1.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_s_c\=1.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n5_mul_s_c\=10.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_s_c\=10.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n5_mul_s_c\=100.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_s_c\=100.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n5_mul_s_c\=1000.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_s_c\=1000.0_bellman.h5 --gpu

# mul_s features r0.15, n5, c 1.0->1000.0
python eval.py --resultfile ../data/results/results_Sep18_r0.15n5_mul_s_c\=1.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_s_c\=1.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.15n5_mul_s_c\=10.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_s_c\=10.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.15n5_mul_s_c\=100.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_s_c\=100.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.15n5_mul_s_c\=1000.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_s_c\=1000.0_bellman.h5 --gpu

# mul_s features r0.1, n7, c 1.0->1000.0
python eval.py --resultfile ../data/results/results_Sep18_r0.1n7_mul_s_c\=1.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_s_c\=1.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n7_mul_s_c\=10.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_s_c\=10.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n7_mul_s_c\=100.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_s_c\=100.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n7_mul_s_c\=1000.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type mul_s --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_mul_s_c\=1000.0_bellman.h5 --gpu

### BASE FEATURES ###

# base features r0.1, n5, c 1.0->1000.0
python eval.py --resultfile ../data/results/results_Sep18_r0.1n5_base_c\=1.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type base --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_base_c\=1_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n5_base_c\=10.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type base --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_base_c\=10.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n5_base_c\=100.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type base --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_base_c\=100.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n5_base_c\=1000.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type base --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_base_c\=1000.0_bellman.h5 --gpu

# base features r0.15, n5, c 1.0->1000.0
python eval.py --resultfile ../data/results/results_Sep18_r0.15n5_base_c\=1.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type base --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_base_c\=1_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.15n5_base_c\=10.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type base --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_base_c\=10.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.15n5_base_c\=100.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type base --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_base_c\=100.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.15n5_base_c\=1000.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type base --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_base_c\=1000.0_bellman.h5 --gpu

# base features r0.1, n7, c 1.0->1000.0
python eval.py --resultfile ../data/results/results_Sep18_r0.1n7_base_c\=1.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type base --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_base_c\=1_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n7_base_c\=10.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type base --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_base_c\=10.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n7_base_c\=100.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type base --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_base_c\=100.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n7_base_c\=1000.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type base --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_base_c\=1000.0_bellman.h5 --gpu


### LANDMARK FEATURES ###
# landmark features r0.1, n5, c 1.0->1000.0
python eval.py --resultfile ../data/results/results_Sep18_r0.1n5_landmark_c\=1.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type landmark --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_landmark_c\=1.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n5_landmark_c\=10.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type landmark --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_landmark_c\=10.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n5_landmark_c\=100.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type landmark --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_landmark_c\=100.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n5_landmark_c\=1000.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n5.h5 feature finger bij --feature_type landmark --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_landmark_c\=1000.0_bellman.h5 --gpu

# landmark features r0.1, n7, c 1.0->1000.0
python eval.py --resultfile ../data/results/results_Sep18_r0.1n7_landmark_c\=1.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type landmark --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_landmark_c\=1.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n7_landmark_c\=10.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type landmark --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_landmark_c\=10.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n7_landmark_c\=100.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type landmark --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_landmark_c\=100.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.1n7_landmark_c\=1000.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type landmark --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_landmark_c\=1000.0_bellman.h5 --gpu

# landmark features r0.15, n5, c 1.0->1000.0
python eval.py --resultfile ../data/results/results_Sep18_r0.15n5_landmark_c\=1.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type landmark --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_landmark_c\=1.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.15n5_landmark_c\=10.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type landmark --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_landmark_c\=10.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.15n5_landmark_c\=100.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type landmark --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_landmark_c\=100.0_bellman.h5 --gpu

python eval.py --resultfile ../data/results/results_Sep18_r0.15n5_landmark_c\=1000.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type landmark --weightfile ../data/weights/Sep15_label_r0.15_n7_cleanfilt_landmark_c\=1000.0_bellman.h5 --gpu
'''
# TIMESTEP FEATURES
# timestep features r0.1, n7, c1.0->1000.0
python  eval.py --resultfile ../data/results/results_Sep18_r0.1n7_timestep_c\=1.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type timestep --weightfile ../data/weights/Sep18_final_r0.15n7_timestep_c\=1.0_bellman.h5 --gpu

python  eval.py --resultfile ../data/results/results_Sep18_r0.1n7_timestep_c\=10.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type timestep --weightfile ../data/weights/Sep18_final_r0.15n7_timestep_c\=10.0_bellman.h5 --gpu

python  eval.py --resultfile ../data/results/results_Sep18_r0.1n7_timestep_c\=100.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type timestep --weightfile ../data/weights/Sep18_final_r0.15n7_timestep_c\=100.0_bellman.h5 --gpu

python  eval.py --resultfile ../data/results/results_Sep18_r0.1n7_timestep_c\=1000.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 feature finger bij --feature_type timestep --weightfile ../data/weights/Sep18_final_r0.15n7_timestep_c\=1000.0_bellman.h5 --gpu



# timestep features r0.15, n5, c1.0->1000.0
python  eval.py --resultfile ../data/results/results_Sep18_r0.15n5_timestep_c\=1.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type timestep --weightfile ../data/weights/Sep18_final_r0.15n7_timestep_c\=1.0_bellman.h5 --gpu

python  eval.py --resultfile ../data/results/results_Sep18_r0.15n5_timestep_c\=10.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type timestep --weightfile ../data/weights/Sep18_final_r0.15n7_timestep_c\=10.0_bellman.h5 --gpu

python  eval.py --resultfile ../data/results/results_Sep18_r0.15n5_timestep_c\=100.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type timestep --weightfile ../data/weights/Sep18_final_r0.15n7_timestep_c\=100.0_bellman.h5 --gpu

python  eval.py --resultfile ../data/results/results_Sep18_r0.15n5_timestep_c\=1000.0_bellman.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n5.h5 feature finger bij --feature_type timestep --weightfile ../data/weights/Sep18_final_r0.15n7_timestep_c\=1000.0_bellman.h5 --gpu






