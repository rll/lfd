# Figure 8 Cross Validation
# Baseline on r0.1, r0.15, n5, n7
python -i eval.py --resultfile ../data/results/results_fig8_Sep18_r0.1n5_greedy.h5 eval --ground_truth 0 ../data/misc/fig8_finalds.h5  ../data/misc/Sep19_fig8_r0.1_n5.h5 greedy finger bij --fake_data_segment demo3_seg00_Sep09 --num_steps 6 --gpu --downsample_size 0.03

#python eval.py --resultfile ../data/results/results_fig8_Sep18_r0.1n7_greedy.h5 eval --ground_truth 0 ../data/misc/fig8_finalds.h5 ../data/misc/Sep19_fig8_r0.1_n7.h5 greedy finger bij --fake_data_segment demo3_seg00_Sep09 --num_steps 6 --gpu

#python eval.py --resultfile ../data/results/results_fig8_Sep18_r0.15n5_greedy.h5 eval --ground_truth 0 ../data/misc/fig8_finalds.h5 ../data/misc/Sep19_fig8_r0.15_n5.h5 greedy finger bij --fake_data_segment demo3_seg00_Sep09 --num_steps 6 --gpu



