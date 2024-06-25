# Dataset
- Using the [GRIT dataset](https://huggingface.co/datasets/zzliang/GRIT) from huggingface.
    
    - step 1: download the raw parquet file
    ```bash
    wget https://huggingface.co/datasets/zzliang/GRIT/resolve/main/grit-20m/coyo_0_snappy.parquet
    ```
    - step 2: download the raw image using `img2dataset`
    ```bash
    img2dataset --url_list . --input_format parquet\
    --url_col "url" --caption_col "caption" --output_format webdataset \
    --output_folder grit_raw --processes_count 4 --thread_count 64 --image_size 112 \
    --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
    --enable_wandb False
    
    ```