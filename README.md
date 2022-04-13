## 1 介绍

本文介绍YOLO v5 GPU 训练 & 推理自定义数据集的方法。



## 2 YOLO v5环境配置

- Python 3.8
- PyTorch 1.8

```shell
conda activate yolo5
source activate yolo5
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts

pip install pyyaml tqdm opencv-python pandas requests matplotlib scikit-learn seaborn tensorboard
```

## 3 数据集格式

数据集文件夹内的文件格式如下(假设数据集和yolov5 repo文件夹在同一个级别，叫`mlzoo-config-new`，那么在yolov5文件夹访问他的路径就是`../mlzoo-config-new`)：

- `images`: 存放每张训练图片，名字为`id.jpg`，当日也支持其他格式

- `labels`: 存放每个图片的annotation，每个label文件和image的名字对应，叫做`id.txt`，格式为`label x y h w`其中`x y h w`分别是中心x，中心y，高和宽，他们都被归一化了。例子：

  ```
  1 0.428000 0.293333 0.424000 0.266667
  1 0.609000 0.684000 0.306000 0.632000
  1 0.404000 0.534667 0.336000 0.568000
  ```

- `train.txt`: 训练集列表，每行是一个图片的绝对链接(`images`里面的图片)

- `valid.txt`: 验证集列表，每行是一个图片的绝对链接 (`images`里面的图片)

- `config.yaml`：训练设置，设置如下

  ```yaml
  # base 目录
  path: ../mlzoo-config-new
  # 之前说的两个文件，也在这个目录
  train: train.txt
  val: valid.txt
  cache_path: ../mlzoo-config-new/
  # number of classes
  nc: 2
  
  # 类别0为cat，类别1为dog
  names: ['cat', 'dog']
  
  ```

- `mask_yolov5xxxx.yaml`:模型参数配置，可以从yolov5官方repo的`models`路径下找到



## 4 训练例子 猫狗

接下来在Pascal VOC数据集中，只抽取出包含猫和狗的图片，并转换为YOLO输入格式

1. 用PyTorch下载Pascal VOC数据集

```python
import torch
import torchvision

torchvision.datasets.VOCDetection(root='.', year='2012', image_set='train', download=True)
```

2. 基于[Pascal VOC 转换 YOLO格式](http://mlzoo.cn/index.php/2022/04/12/pascal-voc-to-yolo-format.html)中的方法，将Pascal VOC格式转换为YOLO格式，编辑`classes.names`，选择要抽取的label：

```
%%writefile classes.names
cat
dog
```

运行以下代码

```python
#coding:utf-8
from __future__ import print_function
 
import os
import random
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm_notebook as tqdm
import shutil
annotation_list = []

def xml_reader(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    size = tree.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)
    return width, height, objects


def voc2yolo(filename, target_path):
    try:
        os.mkdir(target_path)
    except:
        pass
    
    classes_dict = {}
    with open("classes.names") as f:
        for idx, line in enumerate(f.readlines()):
            class_name = line.strip()
            classes_dict[class_name] = idx
    
    width, height, objects = xml_reader(filename)

    lines = []
    for obj in objects:
        x, y, x2, y2 = obj['bbox']
        class_name = obj['name']
        
        ### Test
        if class_name not in classes_dict:
            continue
        
        
        label = classes_dict[class_name]
        
        cx = (x2+x)*0.5 / width
        cy = (y2+y)*0.5 / height
        w = (x2-x)*1. / width
        h = (y2-y)*1. / height
        line = "%s %.6f %.6f %.6f %.6f\n" % (label, cx, cy, w, h)
        lines.append(line)
    
    
    
    txt_name = os.path.join(target_path, filename.replace(".xml", ".txt").replace("labels_voc", "labels").split('/')[-1])
    
    ### Test
    if len(lines) > 0:
        annotation_list.append(filename.split('/')[-1].split('.')[0].replace(".xml", ""))
        with open(txt_name, "w") as f:
            f.writelines(lines)


def get_image_list(image_dir, target_path, suffix=['jpg', 'jpeg', 'JPG', 'JPEG','png']):
    
    try:
        os.rmdir(target_path)
        print('Removed existing image path')
    except:
        pass
    try:
        os.mkdir(target_path)
        print('Created target path:', target_path)
    except:
        pass
    
    '''get all image path ends with suffix'''
    if not os.path.exists(image_dir):
        print("PATH:%s not exists" % image_dir)
        return []
    imglist = []
    
    ins = 0
    alls = 0
    
    for root, sdirs, files in os.walk(image_dir):
        if not files:
            continue
        for filename in files:
            if filename.split('.')[0] not in annotation_list:
                continue
            
            filepath = os.path.join(os.getcwd(), root, filename)
            target_file = os.path.join(target_path, filename)
            
            if filename.split('.')[-1] in suffix:
                imglist.append(filepath + '\n')
                shutil.copy(filepath, target_file)
    return imglist


def imglist2file(imglist, target_path):
    
    imglist = [os.path.join(os.getcwd(), target_path, 'images', item.split('/')[-1]) for item in imglist ]
    random.shuffle(imglist)
    train_list = imglist[:-len(imglist) // 4]
    valid_list = imglist[-len(imglist) // 4:]
    
    print('train_list size:', len(train_list))
    print('valid_list size:', len(valid_list))
    
    with open(os.path.join(target_path, 'train.txt'), 'w') as f:
        f.writelines(train_list)
    with open(os.path.join(target_path, 'valid.txt'), 'w') as f:
        f.writelines(valid_list)
        
def transform_and_save(annotation_path, image_path, target_path):
    
    annotation_target = os.path.join(target_path, 'labels')
    image_target = os.path.join(target_path, 'images')
    try:
        shutil.rmtree(target_path,ignore_errors=True) 
        print('Target path existed, deleted')
    except:
        pass
    os.mkdir(target_path)
    os.mkdir(annotation_target)
    os.mkdir(image_target)
    
    xml_path_list = glob.glob(os.path.join(annotation_path, '*.xml'))
    
    for xml_path in tqdm(xml_path_list):
        voc2yolo(xml_path, annotation_target)
    
    imglist = get_image_list(image_path, image_target)
    imglist2file(imglist, target_path)

anno = 'VOCdevkit/VOC2012/Annotations'
imgs = 'VOCdevkit/VOC2012/JPEGImages'
target_path = 'mlzoo-config-new'

transform_and_save(anno, imgs, target_path)
```

处理好的图片会被放到`mlzoo-config-new`文件夹。

3. 用notebook编辑`config.yaml`

```python
%%writefile mlzoo-config-new/config.yaml
# base 目录
path: ../mlzoo-config-new
# 之前说的两个文件，也在这个目录
train: train.txt
val: valid.txt
cache_path: ../mlzoo-config-new/
# number of classes
nc: 2

# 类别0为cat，类别1为dog， 对应classes.names
names: ['cat', 'dog']
```

4. 下载yolov5 repo, 并进入yolov5目录

```shell
git clone https://github.com/ultralytics/yolov5
cd yolov5
```

5. 复制`models/yolov5s.yaml`到`../mlzoo-config-new/`

```python
models/yolov5s.yaml ../mlzoo-config-new/
```

6. 从头训练YOLO模型

```shell
python train.py --data ../mlzoo-config-new/config.yaml --cfg ../mlzoo-config-new/yolov5s.yaml  --epoch 100 --batch-size 4 --device 0
```

7. 下载预训练的[YOLOv5模型](https://github.com/ultralytics/yolov5/releases)，并做迁移学习

```shell
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt

python train.py --data ../mlzoo-config-new/config.yaml --cfg ../mlzoo-config-new/yolov5s.yaml  --epoch 100 --batch-size 4 --device 0 --weights ./yolov5s.pt
```

训练结果会在`runs/exp[运行号]`下面存储。

![](https://bastudypic.oss-cn-hongkong.aliyuncs.com/auto-upload/2022/04/image-20220413181215080.png)

8. 推理

```shell
# 摄像头
python detect.py --weights runs/train/exp[运行号]/weights/best.pt --source 0 --device 0

# 图片
python detect.py --weights runs/train/exp[运行号]/weights/best.pt --source xxx.jpg --device 0

# 视频
python detect.py --weights runs/train/exp[运行号]/weights/best.pt --source xxx.mp4 --device 0

# 检测YouTube网络视频
python detect.py --weights runs/train/exp[运行号]/weights/best.pt --source https://youtu.be/xxx --device 0


# 检测RTSP流视频
python detect.py --weights runs/train/exp[运行号]/weights/best.pt --source rtsp://xxx.mp4 --device 0
```



![](https://bastudypic.oss-cn-hongkong.aliyuncs.com/auto-upload/2022/04/image-20220413180623168.png)

