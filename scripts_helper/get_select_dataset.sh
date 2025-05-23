python scripts_helper/encode_ave.py \
    --folder_path dataset/data/AVE/videos/ \
    --feature_path dataset/data/AVE/feature \
    --ave_labels dataset/data/AVE/data/labels.h5 \
    --ave_annotations dataset/data/AVE/data/Annotations.txt \
    --ave_meta dataset/data/AVE/data/meta_data.csv \
    --ave_category dataset/data/AVE/data/categorys.txt \
    --ave_feature_root dataset/feature/select/ave/ \


# 修改avepm的label得到S-PM
python scripts_helper/modify_avepm.py \
    --selected_csv_file dataset/csvfiles/ave_pm_selected_files.csv \
    --mapping_file dataset/csvfiles/pm_selected_category.csv \
    --src_feat_path dataset/feature/features \
    --dst_feat_path dataset/feature/select/avepm \

