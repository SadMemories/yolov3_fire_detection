# yolov3_spp_fire模型记录

**前提：此批次训练将不使用预训练权重进行训练，训练是在13063张图片中进行 这些图片中包括了已混淆图片**

#### train1：

   未改变anchor以及未加入SE及CBAM之前进行的训练，9个anchor分别为：10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
   训练时具体参数：

​                   epochs: 235
​                   batch-size: 4
​                   cfg: cfg/yolov3-spp.cfg
​                   data: data/fire.data
​                   weights: 空

*val result   Map@0.5:65.5%*

*test result Map@0.5:67.2%*

#### train2:

   利用kmeans算法聚类除了anchor，未加入SE及CBAM，9个anchor分别为：10,10,  16,14,  33,50,  30,17,  62,66,  59,35,  116,194,  156,163,  373,447
   kmeans算法聚类得到的width height height/width分别如下：

| width    | height   | height/width |
| -------- | -------- | :----------: |
| 10.000   | 12.000   |     1.2      |
| 22.000   | 23.000   |     1.0      |
| 31.000   | 52.000   |     1.7      |
| 59.000   | 34.000   |     0.6      |
| 72.000   | 77.000   |     1.1      |
| 174.000  | 96.000   |     0.6      |
| 108.000  | 164.000  |     1.5      |
| 276.000  | 251.000  |     0.9      |
| 1780.000 | 1856.000 |     1.0      |
|          |          |              |

 训练时具体参数：

​                   epochs: 235
​                   batch-size: 4
​                   cfg: cfg/yolov3-spp.cfg
​                   data: data/fire.data
​                   weights: 空

*test result Map@0.5:66.3%*

#### train3:

 未使用kmeans算法聚类，加入了SE模块

训练时具体参数：

​                epochs:235

​                batch-size:4       

​                cfg:cfg/yolov3-spp.cfg

​                data:data/fire.data

​                weights:空

*val result Map@0.5: 66.1%*

