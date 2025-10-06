
## Environment setup
**Classification environment:** 
We recommend installing the required packages by running the command:
```sh
pip install -r requirements.txt
```


## Getting the data
<!-- 
Download dataset from 
[CNNDetection CVPR2020 (Table1 results)](https://github.com/peterwang512/CNNDetection), 
[GANGen-Detection (Table2 results)](https://github.com/chuangchuangtan/GANGen-Detection) 
([googledrive](https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj?usp=sharing)), 
[UniversalFakeDetect CVPR2023](https://github.com/Yuheng-Li/UniversalFakeDetect) 
([googledrive](https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-?usp=drive_link)), 
[DIRE 2023ICCV](https://github.com/ZhendongWang6/DIRE) 
([googledrive](https://drive.google.com/drive/folders/1jZE4hg6SxRvKaPYO_yyMeJN_DOcqGMEf?usp=sharing)), 
Diffusion1kStep [googledrive](https://drive.google.com/drive/folders/14f0vApTLiukiPvIHukHDzLujrvJpDpRq?usp=sharing).
-->
|                        | paper  | Url  |
|:----------------------:|:-----:|:-----:|
| Train set              | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)                   | [Baidudrive](https://pan.baidu.com/s/1l-rXoVhoc8xJDl20Cdwy4Q?pwd=ft8b)                 | 
| Val   set              | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)                   | [Baidudrive](https://pan.baidu.com/s/1l-rXoVhoc8xJDl20Cdwy4Q?pwd=ft8b)                 | 
| Table1 Test            | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)                   | [Baidudrive](https://pan.baidu.com/s/1l-rXoVhoc8xJDl20Cdwy4Q?pwd=ft8b)                 | 
| Table2 Test            | [FreqNet AAAI2024](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection)        | [googledrive](https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj?usp=sharing)   | 
| Table3 Test            | [UniversalFakeDetect CVPR2023](https://github.com/Yuheng-Li/UniversalFakeDetect)        | [googledrive](https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-?usp=sharing)| 


```
pip install gdown==4.7.1

chmod 777 ./download_dataset.sh

./download_dataset.sh
```
## Directory structure
<details>
<summary> Click to expand the folder tree structure. </summary>

```
datasets
|-- ForenSynths_train_val
|   |-- train
|   |   |-- car
|   |   |-- cat
|   |   |-- chair
|   |   `-- horse
|   `-- val
|   |   |-- car
|   |   |-- cat
|   |   |-- chair
|   |   `-- horse
|   |-- test
|       |-- biggan
|       |-- cyclegan
|       |-- deepfake
|       |-- gaugan
|       |-- progan
|       |-- stargan
|       |-- stylegan
|       `-- stylegan2
`-- Generalization_Test
    |-- ForenSynths_test       # Table1
    |   |-- biggan
    |   |-- cyclegan
    |   |-- deepfake
    |   |-- gaugan
    |   |-- progan
    |   |-- stargan
    |   |-- stylegan
    |   `-- stylegan2
    |-- GANGen-Detection     # Table2
    |   |-- AttGAN
    |   |-- BEGAN
    |   |-- CramerGAN
    |   |-- InfoMaxGAN
    |   |-- MMDGAN
    |   |-- RelGAN
    |   |-- S3GAN
    |   |-- SNGAN
    |   `-- STGAN
    `-- UniversalFakeDetect # Table3
    |   |-- dalle
    |   |-- glide_100_10
    |   |-- glide_100_27
    |   |-- glide_50_27
    |   |-- guided          # Also known as ADM.
    |   |-- ldm_100
    |   |-- ldm_200
    |   `-- ldm_200_cfg
   


```
</details>

## Training the model 
```sh
export CUDA_VISIBLE_DEVICES=0 && nohup python train_vit.py --name 4class-resnet-car-cat-chair-horse --dataroot ./datasets/ForenSynths_train_val --classes car,cat,chair,horse --batch_size 32 --delr_freq 20 --lr 0.0004 --niter 50 

```

## Testing the detector
Modify the dataroot in test.py.
```sh
export CUDA_VISIBLE_DEVICES=0 && nohup python test_vit.py  --model_path ./test.pth  --batch_size {BS}
```
