# Training ACT

Taking `real-gaps_stata_v1` as an example

1. Navigate to `dog-park/lucidsim_experiments/datasets/act_depth/real-gaps_stata_v1/sweep.py` to run the sweep file and then run the `launch.py` file to upload jobs

2. Run `launch_rgb_teacher_node.py` to start collecting trajectories. After this is donem proceed to 3.

3. Run `act_trainer.py` to start training with ACT policy. 

# Notes:

1. Make sure the queue names are aligned and keep the `launch.py` running for the whole trainning.
