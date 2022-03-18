# Dense Depth Estimation from Multiple 360-degree Images Using Virtual Depth
[[Paper]](https://link.springer.com/article/10.1007/s10489-022-03391-w) [[arXiv]](https://arxiv.org/abs/2112.14931)

This is the official code of our APIN 2022 paper **"Dense Depth Estimation from Multiple 360-degree Images Using Virtual Depth"**.

## Prerequisites
- Ubuntu 18.04
- C++11 Compiler
- OpenCV > 3.0 (**Tested with OpenCV 3.4.5.**)

## Usage
Clone the repository:
```
git clone <>
```
You can simply execute `build.sh` to build this program.
```
cd 360depth-demo
chmod +x build.sh
./build.sh  
```
This will create *libDEMO.so* at *lib* folder and two executable *image* and *video* in current folder.  

You can simply run demo like below.
```
./image [the number of cameras] [max depth] [interval] [save folder path] [0th images folder] [1st images folder] ...
```

## Results
| Data Set  | *classroom* |            | *smallroom* |            |
|:----------|:-----------:|:----------:|:-----------:|:----------:|
|           |    MSE↓     |   PSNR↑    |    MSE↓     |   PSNR↑    |
| GC-Net    |    0.951    |   20.239   |    5.801    |   12.366   |
| PSMNet    |    4.127    |   13.844   |    7.862    |   11.045   |
| GA-Net    |    2.346    |   16.298   |    4.581    |   13.391   |
| 360SD-Net |    0.218    |   26.625   |    0.581    |   22.361   |
| BiFuse    |    1.803    |   17.481   |    4.108    |   13.866   |
| UniFuse   |    0.215    |   26.707   |    1.655    |   17.825   |
| **ours**  |  **0.193**  | **27.178** |  **0.303**  | **25.193** |