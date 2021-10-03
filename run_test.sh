python test_models.py --model PDCNet --pre_trained_model megadepth \
    --path_query_image FTDataset/20210817_for_testing/zed_left_depth/depth_0000.png \
    --path_reference_image FTDataset/20210817_for_testing/hua_img/img_00000.jpg \
    --write_dir evaluation/ \
    --pre_trained_models_dir /srv/beegfs02/scratch/efficient_nn_mobile/data/dense_matching/pretrained_models/ \
    PDCNet \
    --multi_stage_type multiscale_homo_from_quarter_resolution_uncertainty \
    --mask_type proba_interval_1_above_10
