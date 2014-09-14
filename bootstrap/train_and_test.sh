#! /bin/bash
python bootstrap/bootstrapping.py --train_sizes 10 30 --alpha 100 derived-traj
python bootstrap/eval.py --train_sizes 10 30 --i_end 10 eval bigdata/bootstrap/bootstrap_res.h5 bigdata/misc/sept_13_0.1_test.h5 bij