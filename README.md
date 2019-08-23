# AlexNet_tensorflow
## 训练网络
17flowers:
```
python alexnet_train.py --train_data_path="训练数据地址" --val_data_path="测试数据地址" --data_nums=数据总量
```
cifar10:
```
python alexnet_train.py --train_data_path="训练数据地址" --val_data_path="测试数据地址" --data_nums=数据总量 --num_classes=10 --dataset="cifar10" --image_size=32
```

## 去过拟合 

## 冻结网络
冻结model的例子17flowers的model:
```
python freeze_model2pb.py
```
使用pb文件:
```
python use_pb_detect.py
```

## 使用TensorRT
命令行pb转uff:
```
convert-to-uff tensorflow -o="model.uff" --input-file="model.pb" --O="outputdata"
```
使用uff:
```
```
