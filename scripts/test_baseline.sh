# Scripts to run eval on different levels of difficulty

echo RUNNING Baseline
# no features
python eval.py --animation 0 --i_end 100 --resultfile ../data/results/Sep13_r0.15_n7_no_feature.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n7.h5 greedy finger bij

python eval.py --animation 0 --i_end 100 --resultfile ../data/results/Sep13_r0.15_n9_no_feature.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.15_n9.h5 greedy finger bij

python eval.py --animation 0 --i_end 100 --resultfile ../data/results/Sep13_r0.1_n7_no_feature.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n7.h5 greedy finger bij

python eval.py --animation 0 --i_end 100 --resultfile ../data/results/Sep13_r0.1_n9_no_feature.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.1_n9.h5 greedy finger bij

python eval.py --animation 0 --i_end 100 --resultfile ../data/results/Sep13_r0.2_n7_no_feature.h5 eval --ground_truth 0 ../bigdata/misc/overhand_actions.h5 ../data/misc/Sep13_r0.2_n7.h5 greedy finger bij


