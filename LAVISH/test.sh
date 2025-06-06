model_path = ""

python3 main_trans.py \
        --Adapter_downsample=8 \
        --audio_folder=/dataset/data/raw_audio \
        --video_folder=/dataset/data/video_frames \
        --is_audio_adapter_p1=1 \
        --is_audio_adapter_p2=1 \
        --is_audio_adapter_p3=0 \
        --is_before_layernorm=1 \
        --is_bn=1 \
        --is_fusion_before=1 \
        --is_gate=1 \
        --is_post_layernorm=1 \
        --is_vit_ln=0 \
        --num_conv_group=2 \
        --num_tokens=2 \
        --num_workers=16 \
        --is_multimodal=1 \
        --vis_encoder_type=swin \
        --mode=test \
        --avepm=true \
        --model_save_dir=$model_path \