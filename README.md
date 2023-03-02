# MyNetworks

1、单标签分类任务。

** models
model_LeNet.py 
model_AlexNet.py
model_VGG.py
model_ResNet.py
model_MobileNetV2.py
model_MobileNetV3.py
model_DenseNet.py
model_EfficientNet.py
model_EfficientNetV2.py
model_googlenet.py
model_MobileViT.py
model_RegNet.py
model_ShuffleNetV2.py

** singlelabel_train.py


## singlelabel_train.py

## pth_predict_single.py

## pth_predict_bacth.py

## pth2onnx.py

## onnx_predict_single.py

## onnx_predict_batch.py


















    # # load pretrain weights
    # # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    # model_weight_path = "D:\\MyNetworks\\pretrain\\mobilenet_v2-b0353104.pth"
    # assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    # pre_weights = torch.load(model_weight_path, map_location='cpu')

    # ## delete classifier weights：由于修改了全连接层的节点数，因此和预训练模型的维度不一样，因此就可以把不一致的给删除
    # ## missing_keys,unexpected_keys
    # ## strict=False
    # ###### True 时，代表有什么要什么，每一个键都有。
    # ###### False 时，有什么我要什么，没有的不勉强。
    # ############## missing_keys, unexpected_keys，返回值：缺失的键，不期望的键。
    # pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    # # print(pre_dict)
    # missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
    # # print(missing_keys, unexpected_keys)

    # # freeze features weights   冻结"特征"权重
    # for param in net.features.parameters():
    #     param.requires_grad = True



    ## 打印的是：模块名字.序号.权重名（注意此处不会打印relu，pool不需要back的层
    ## 如果直接打印param, 即 print(name,param), 打印结果：打印出来的详细参数
    # for name, param in net.named_parameters():
    #     print(name, '\t', param.size())
    # for param in net.parameters():
    #     print(param.requires_grad)
    # for name, param in net.named_parameters():
    #     print(name, '\t', param.requires_grad)
    # for name, param in net.named_parameters():
    #     if param.requires_grad:
    #         print(name)