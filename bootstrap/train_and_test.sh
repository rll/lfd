#! /bin/bash
python bootstrap/bootstrapping.py --train_sizes 50 100 --alpha 100 derived-traj bigdata/bootstrap/bootstrap_res_0.15_0.h5 bigdata/bootstrap/sept_13_0.15_train_0.h5 
python bootstrap/eval.py --train_sizes 50 100 --i_end 100 eval bigdata/bootstrap/bootstrap_res_0.15_0.h5 bigdata/misc/sept_13_0.15_test.h5 bij