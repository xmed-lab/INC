Few-Shot Lymph Node Metastasis Classification Meets High Performance on Whole Slide Images via the Informative Non-Parametric Classifier

## Introduction
This is a experimental code for the Informative Non-Parametric Classifier.
The paper is published on MICCAI 2024 at https://papers.miccai.org/miccai-2024/paper/2187_paper.pdf

## Install
```
pip install -r requirements.txt
sudo apt install libvips-dev
```

## Prepared Dataset
1.Download CAMELYON16 dataset to ~/CAMELYON16
2.Convert WSI to patches (x40 svs file to x20 jpeg files)
```
python prepare/wsi_to_patch.py ~/CAMELYON16/images/ ~/CAMELYON16/patch/images 10
```
3.Convert xml annotations to GT mask
```
python prepare/xml_to_gt_camelyon.py ~/CAMELYON16/annotations/ ~/CAMELYON16/patch/gt/ ~/CAMELYON16/images/ 512
```
4.Prepare dataset information json
```
python prepare/make_data_info_camelyon.py ~/CAMELYON16/patch/gt/ data_info/camelyon16.json
```

## Download pretrained weights
```
wget https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights/dino_vit_small_patch8_ep200.torch
mv dino_vit_small_patch8_ep200.torch ckpt/
```

# Run INC
1.Extract features (--nproc_per_node control gpu num)
```
python -m torch.distributed.launch --nproc_per_node=8 core/wsi_feature_extractor.py --data_path ~/CAMELYON16/patch/images/ --pretrained_weights ckpt/dino_vit_small_patch8_wsi.torch --patch_size 8 --dump_features ~/CAMELYON16/patch/feat_dino_vits8_wsi/ --num_workers 8 --batch_size_per_gpu 64
```
2.Load features and run INC
```
python -u core/main.py --mode default --topk 40 --temperature 10 --related_thresh 0.88 --gallery_num 8 --raw_feature_path ~/CAMELYON16/patch/feat_dino_vits8_wsi --wsi_path ~/CAMELYON16/images --dump_features ~/CAMELYON16/patch/output_dino_vits8_wsi --dataset_info data_info/camelyon16.json &> INC_8shot_results.txt
```

Note: CAMELYON16-C is implemented from https://github.com/superjamessyx/robustness_benchmark to simulate WSI corruptions.
Uncommand Line 93 of core/wsi_feature_extractor.py to activate the RandomDistortions() for CAMELYON16-C.
Then run command 1 with another --dump_features, and run command 2 with corresponding --raw_feature_path --dump_features --dataset_info.

For CAMELYON17, items in data_info/camelyon17.json with "fixed_test_set": true are the test WSIs.
CAMELYON17 use the annotations of CAMELYON16, which have been included in the data_info/camelyon17.json.
Run command 1 & 2 by replacing all CAMELYON16 to CAMELYON17 after preparing the dataset by prepare/wsi_to_patch.py.

# Citation
@InProceedings{ Li_MICCAI2024,
   author = { Li, Yi and Zhang, Qixiang and Xiang, Tianqi and Lin, Yiqun and Zhang, Qingling and Li, Xiaomeng },
   title = { Few-Shot Lymph Node Metastasis Classification Meets High Performance on Whole Slide Images via the Informative Non-Parametric Classifier }, 
   booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},
   year = {2024},
   publisher = {Springer Nature Switzerland}
   volume = { 12 }
   month = {October}, 
}
