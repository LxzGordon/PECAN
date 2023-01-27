#train partner
for i in {9015..9015}
do
export PBT_DATA_DIR="pbt_data_dir/" && python pbt/pbt_model_pool_entropy_parallel.py with fixed_mdp layout_name="simple" EX_NAME="pbt_simple" TOTAL_STEPS_PER_AGENT=1.1e7 REW_SHAPING_HORIZON=5e6 LR=8e-4 GPU_ID=0 POPULATION_SIZE=5 SEEDS="[$i]" NUM_SELECTION_GAMES=6 VF_COEF=0.5 MINIBATCHES=10 TIMESTAMP_DIR=False ENTROPY_POOL=0.01 ENT_VERSION=3
done
#train ego
for i in {8015..8015}
do
export PBT_DATA_DIR="pbt_data_dir_2/" && python pbt/pbt_model_pool.py with fixed_mdp layout_name="simple" EX_NAME="pbt_simple" TOTAL_STEPS_PER_AGENT=1.1e7 REW_SHAPING_HORIZON=5e6 LR=8e-4 GPU_ID=0 POPULATION_SIZE=15 SEEDS="[3]" NUM_SELECTION_GAMES=6 VF_COEF=0.5 MINIBATCHES=10 TIMESTAMP_DIR=False ENTROPY_POOL=0.0 PRIORITIZED_SAMPLING=True ALPHA=3.0 METRIC=1.0 LOAD_FOLDER_LST="pbt_data_dir/pbt_simple/seed_9015/agent0/pbt_iter1/:pbt_data_dir/pbt_simple/seed_9015/agent1/pbt_iter1/:pbt_data_dir/pbt_simple/seed_9015/agent2/pbt_iter1/:pbt_data_dir/pbt_simple/seed_9015/agent3/pbt_iter1/:pbt_data_dir/pbt_simple/seed_9015/agent4/pbt_iter1/:pbt_data_dir/pbt_simple/seed_9015/agent0/pbt_iter152/:pbt_data_dir/pbt_simple/seed_9015/agent1/pbt_iter152/:pbt_data_dir/pbt_simple/seed_9015/agent2/pbt_iter152/:pbt_data_dir/pbt_simple/seed_9015/agent3/pbt_iter152/:pbt_data_dir/pbt_simple/seed_9015/agent4/pbt_iter152/:pbt_data_dir/pbt_simple/seed_9015/agent0/pbt_iter305/:pbt_data_dir/pbt_simple/seed_9015/agent1/pbt_iter305/:pbt_data_dir/pbt_simple/seed_9015/agent2/pbt_iter305/:pbt_data_dir/pbt_simple/seed_9015/agent3/pbt_iter305/:pbt_data_dir/pbt_simple/seed_9015/agent4/pbt_iter305/"
done 