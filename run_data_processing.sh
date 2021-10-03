#!/bin/bash
#SBATCH --output=/home/tfausto/log/data_processing_%j.out
#SBATCH --gres=gpu:1
#SBATCH --constraint='titan_xp|geforce_gtx_titan_x'
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
# conda activate dense_matching_env
python -u data_processing.py --model PDCNet --pre_trained_model megadepth \
    --data_path /home/tfausto/code/DenseMatching/FTDataset/20210826\
    --pre_trained_models_dir /srv/beegfs02/scratch/efficient_nn_mobile/data/dense_matching/pretrained_models/ \
    PDCNet \
    --multi_stage_type homography_from_quarter_resolution_uncertainty \
    --mask_type proba_interval_1_above_10
python -u data_processing.py --model PDCNet --pre_trained_model megadepth \
    --data_path /home/tfausto/code/DenseMatching/FTDataset/20210908\
    --pre_trained_models_dir /srv/beegfs02/scratch/efficient_nn_mobile/data/dense_matching/pretrained_models/ \
    PDCNet \
    --multi_stage_type homography_from_quarter_resolution_uncertainty \
    --mask_type proba_interval_1_above_10
python -u data_processing.py --model PDCNet --pre_trained_model megadepth \
    --data_path /home/tfausto/code/DenseMatching/FTDataset/20210909\
    --pre_trained_models_dir /srv/beegfs02/scratch/efficient_nn_mobile/data/dense_matching/pretrained_models/ \
    PDCNet \
    --multi_stage_type homography_from_quarter_resolution_uncertainty \
    --mask_type proba_interval_1_above_10
python -u data_processing.py --model PDCNet --pre_trained_model megadepth \
    --data_path /home/tfausto/code/DenseMatching/FTDataset/20210914\
    --pre_trained_models_dir /srv/beegfs02/scratch/efficient_nn_mobile/data/dense_matching/pretrained_models/ \
    PDCNet \
    --multi_stage_type homography_from_quarter_resolution_uncertainty \
    --mask_type proba_interval_1_above_10
python -u data_processing.py --model PDCNet --pre_trained_model megadepth \
    --data_path /home/tfausto/code/DenseMatching/FTDataset/20210915\
    --pre_trained_models_dir /srv/beegfs02/scratch/efficient_nn_mobile/data/dense_matching/pretrained_models/ \
    PDCNet \
    --multi_stage_type homography_from_quarter_resolution_uncertainty \
    --mask_type proba_interval_1_above_10
python -u data_processing.py --model PDCNet --pre_trained_model megadepth \
    --data_path /home/tfausto/code/DenseMatching/FTDataset/20210915_01\
    --pre_trained_models_dir /srv/beegfs02/scratch/efficient_nn_mobile/data/dense_matching/pretrained_models/ \
    PDCNet \
    --multi_stage_type homography_from_quarter_resolution_uncertainty \
    --mask_type proba_interval_1_above_10
python -u data_processing.py --model PDCNet --pre_trained_model megadepth \
    --data_path /home/tfausto/code/DenseMatching/FTDataset/20210916\
    --pre_trained_models_dir /srv/beegfs02/scratch/efficient_nn_mobile/data/dense_matching/pretrained_models/ \
    PDCNet \
    --multi_stage_type homography_from_quarter_resolution_uncertainty \
    --mask_type proba_interval_1_above_10
python -u data_processing.py --model PDCNet --pre_trained_model megadepth \
    --data_path /home/tfausto/code/DenseMatching/FTDataset/20210917\
    --pre_trained_models_dir /srv/beegfs02/scratch/efficient_nn_mobile/data/dense_matching/pretrained_models/ \
    PDCNet \
    --multi_stage_type homography_from_quarter_resolution_uncertainty \
    --mask_type proba_interval_1_above_10
python -u data_processing.py --model PDCNet --pre_trained_model megadepth \
    --data_path /home/tfausto/code/DenseMatching/FTDataset/20210917_01\
    --pre_trained_models_dir /srv/beegfs02/scratch/efficient_nn_mobile/data/dense_matching/pretrained_models/ \
    PDCNet \
    --multi_stage_type homography_from_quarter_resolution_uncertainty \
    --mask_type proba_interval_1_above_10
