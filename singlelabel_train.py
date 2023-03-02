import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from datasets import __dataNames__
from utils.experiment import adjust_learning_rate, get_logger
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse

from torchsummary import summary
from ptflops import get_model_complexity_info

from models.model_MobileNetV2 import MobileNetV2
from models.model_MobileNetV3 import mobilenet_v3_small
from models.model_MobileNetV3 import mobilenet_v3_large
from models.model_vovnet import vovnet_se_slice

cudnn.benchmark = True

parser = argparse.ArgumentParser(description=" VSEE Network ")

parser.add_argument('--datapath', required=True, help='data path, contain train and test folder')
parser.add_argument('--dataName', required=True, help='data Name')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=4, help='testing batch size')
parser.add_argument('--onnx_batch_size', type=int, default=4, help='onnx batch size')

parser.add_argument('--img_channels', type=int, default=3, help='img channels')
parser.add_argument('--img_height', type=int, default=160, help='img height')
parser.add_argument('--img_width', type=int, default=160, help='img width')

parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')   # "10,12,14,16:2"

parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')

parser.add_argument('--savepthdir', required=True, help='the directory to save pthModel')
parser.add_argument('--saveonnxdir', required=True, help='the directory to save OnnxModel')
parser.add_argument('--savelogdir', required=True, help='the directory to save logs')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default:1)')



def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("using {} device.".format(device))
    
    ## 添加数据路径
    assert os.path.exists(args.datapath), "{} path does not exist.".format(args.datapath)
    train_dataset = datasets.ImageFolder(root=os.path.join(args.datapath, "train"), transform=__dataNames__[args.dataName]["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testdate_dataset = datasets.ImageFolder(root=os.path.join(args.datapath, "train"), transform=__dataNames__[args.dataName]["test"])
    testdate_loader = torch.utils.data.DataLoader(testdate_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # print(train_dataset.classes)             ## classes (list): List of the class names sorted alphabetically.
    # print(train_dataset.class_to_idx)        ## class_to_idx (dict): Dict with items (class_name, class_index).
    # print(train_dataset.imgs)                ## imgs (list): List of (image path, class_index) tuples # 列表：(路径, 类别)
    # print(validate_dataset.classes)          ## classes (list): List of the class names sorted alphabetically.
    # print(validate_dataset.class_to_idx)     ## class_to_idx (dict): Dict with items (class_name, class_index).
    # print(validate_dataset.imgs)             ## imgs (list): List of (image path, class_index) tuples # 列表：(路径, 类别)
    train_num = len(train_dataset)
    test_num = len(testdate_dataset)
    logger.info("using {} images for training, {} images for validation.".format(train_num, test_num))
    
    ## 训练集的图像数量 + 类别数量，测试集的图像数量 + 类别数量
    num_class = len(train_dataset.classes)
    train_per_class_num = list(0 for i in range(num_class))
    for i in range(len(train_dataset.imgs)):
        label = train_dataset.imgs[i][1]
        train_per_class_num[label] += 1
        
    test_per_class_num = list(0 for i in range(num_class))
    for i in range(len(testdate_dataset.imgs)):
        label = testdate_dataset.imgs[i][1]
        test_per_class_num[label] += 1
    for i in range(num_class):
        logger.info('     [Train class %d]  num_total %d' % (i, train_per_class_num[i]))
    logger.info("     --------")
    for i in range(num_class):
        logger.info('     [Test class %d]  num_total %d' % (i, test_per_class_num[i]))
    
    ## 将类别和索引，写入文件json中
    cla_dict = dict((val, key) for key, val in train_dataset.class_to_idx.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open(os.path.join(args.datapath, "class_indices.json"), 'w') as json_file:
        json_file.write(json_str)

    # create model
    net = MobileNetV2(num_classes=num_class)
    # net = vovnet_se_slice()
    
    # net = mobilenet_v3_small(num_classes=num_class)
    # net = mobilenet_v3_large(num_classes=num_class)
    net.to(device)
    
    ## 打印网络信息(输入、输出)和参数大小
    summary(net, input_data = torch.ones(1, args.img_channels, args.img_height, args.img_width))    # C,H,W
    # flops, params = get_model_complexity_info(net, (args.img_channels, args.img_height, args.img_width), 
    #                                           as_strings=True, print_per_layer_stat=True)
    # logger.info("Flops: {}".format(flops))
    # logger.info("Params: " + params)
    

    ## define loss function
    loss_function = nn.CrossEntropyLoss()

    ## construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

    ## load parameters
    start_epoch = 0
    if args.resume:
        # find all checkpoints file and sort according to epoch id
        all_saved_ckpts = [fn for fn in os.listdir(args.savepthdir) if fn.endswith(".pth")]
        all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.savepthdir, all_saved_ckpts[-1])
        print("loading the lastest model in savepthdir: {}".format(loadckpt))
        state_dict = torch.load(loadckpt)
        net.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif args.loadckpt:   # 待测试
        # load the checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt)
        net.load_state_dict(state_dict['model'])
    print("start at epoch {}".format(start_epoch))


    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr, args.lrepochs)
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()   

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, args.epochs, loss)

        # # Traindate
        # net.eval()
        # train_class_correct = list(0. for i in range(num_class))
        # train_class_total = list(0. for i in range(num_class))
        # train_acc = 0.0  # accumulate accurate number / epoch
        # with torch.no_grad():
        #     train_acc_bar = tqdm(train_loader, file=sys.stdout)
        #     for train_acc_data in train_acc_bar:
        #         train_acc_images, train_acc_labels = train_acc_data
        #         outputs = net(train_acc_images.to(device))
        #         predict_train_y = torch.max(outputs, dim=1)[1]
                
        #         for b in range(train_acc_labels.numel()):   # 或 len(images)
        #             label = train_acc_labels[b].item()
        #             if label == predict_train_y[b].item():
        #                 train_class_correct[label] += 1
        #             train_class_total[label] += 1
                
        #         train_acc += torch.eq(predict_train_y, train_acc_labels.to(device)).sum().item()

        #         train_acc_bar.desc = "Train valid epoch[{}/{}]".format(epoch + 1, args.epochs)
        
        
        # train_val_accurate = train_acc / train_num

        # validate
        net.eval()
        class_correct = list(0. for i in range(num_class))
        class_total = list(0. for i in range(num_class))
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(testdate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                
                for b in range(val_labels.numel()):   # 或 len(images)
                    label = val_labels[b].item()
                    if label == predict_y[b].item():
                        class_correct[label] += 1
                    class_total[label] += 1
                
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, args.epochs)
        
        
        val_accurate = acc / test_num
        
        # logger.info('[epoch %d] train_loss: %.3f  train_val_accurate: %.3f val_accuracy: %.3f' % (epoch + 1, running_loss / train_steps, train_val_accurate, val_accurate))
        logger.info('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, running_loss / train_steps, val_accurate))
        # for i in range(num_class):
        #     logger.info('     [Train class %d]  num %.3f total %.3f class_accuracy: %.3f' % 
        #         (i, train_class_correct[i], train_class_total[i], train_class_correct[i]/train_class_total[i]))
        # logger.info("    ----------------------------------------")
        for i in range(num_class):
            logger.info('     [Test class %d]  num %.3f total %.3f class_accuracy: %.3f' % 
                  (i, class_correct[i], class_total[i], class_correct[i]/class_total[i]))

        ##1 保存每一个  
        save_label = str(epoch) + "_" + str(format(val_accurate, ".4f"))
        for i in range(num_class):
            save_label = save_label + "_" + str(format(class_correct[i]/class_total[i], ".3f"))
        
        # save_content = {"epoch": epoch, "model": net.state_dict(), "optimizer": optimizer.state_dict()}
        # torch.save(save_content, args.savepthdir + save_label + ".pth")
        torch.save(net.state_dict(), args.savepthdir + save_label + ".pth")

        dummy_input = torch.randn(args.onnx_batch_size, args.img_channels, args.img_height, args.img_width, requires_grad=True).to(device)
    
        torch.onnx.export(net,                                                         # model being run 
            dummy_input,                                                               # model input (or a tuple for multiple inputs) 
            args.saveonnxdir + save_label + ".onnx",                                   # where to save the model  
            export_params=True,                                                        # store the trained parameter weights inside the model file 
            opset_version=10,                                                          # the ONNX version to export the model to 
            do_constant_folding=True,                                                  # whether to execute constant folding for optimization 
            input_names = ['input1'],                                                  # the model's input names 
            output_names = ['output1'],                                                # the model's output names 
            ) 

    logger.info('Finished Training')


if __name__ == '__main__':
    
    # args = parser.parse_args(["--datapath", "D:\\MyNetworks\\datasets\\chestnut",
    #                         "--dataName", "chestnut",
    #                         "--img_channels", "3",
    #                         "--img_height", "160",
    #                         "--img_width", "160",
    #                         "--batch_size", "32", 
    #                         "--onnx_batch_size", "5",
    #                         "--epochs", "100",
    #                         "--lrepochs", "20,40,60,80:2",
    #                         "--savepthdir", "D:\\MyNetworks\\datasets\\chestnut\\save\\MobileNetV2\\pthRandomSampler\\",
    #                         "--saveonnxdir", "D:\\MyNetworks\\datasets\\chestnut\\save\\MobileNetV2\\onnxRandomSampler\\",
    #                         "--savelogdir", "D:\\MyNetworks\\datasets\\chestnut\\save\\MobileNetV2\\logRandomSampler.txt",])
    
    # args = parser.parse_args(["--datapath", "D:\\MyNetworks\\datasets\\chestnut",
    #                   "--dataName", "chestnut",
    #                   "--img_channels", "3",
    #                   "--img_height", "160",
    #                   "--img_width", "160",
    #                   "--batch_size", "32", 
    #                   "--onnx_batch_size", "5",
    #                   "--epochs", "100",
    #                   "--lrepochs", "20,40,60,80:2",
    #                   "--savepthdir", "D:\\MyNetworks\\datasets\\chestnut\\save\\MobileNetV3_samll\\pth\\",
    #                   "--saveonnxdir", "D:\\MyNetworks\\datasets\\chestnut\\save\\MobileNetV3_samll\\onnx\\",
    #                   "--savelogdir", "D:\\MyNetworks\\datasets\\chestnut\\save\\MobileNetV3_samll\\log.txt",
    #                   "--resum"])
    
    args = parser.parse_args(["--datapath", "D:\\MyNetworks\\datasets\\JunZao",
                      "--dataName", "JunZao",
                      "--img_channels", "3",
                      "--img_height", "160",
                      "--img_width", "160",
                      "--batch_size", "32", 
                      "--onnx_batch_size", "6",
                      "--epochs", "100",
                      "--lrepochs", "20,40,60,80:2",
                      "--savepthdir", "D:\\MyNetworks\\datasets\\JunZao\\save\\MobileNetV2-0225\\pth\\",
                      "--saveonnxdir", "D:\\MyNetworks\\datasets\\JunZao\\save\\MobileNetV2-0225\\onnx\\",
                      "--savelogdir", "D:\\MyNetworks\\datasets\\JunZao\\save\\MobileNetV2-0225\\log.txt",])
    
    # args = parser.parse_args(["--datapath", "D:\\MyNetworks\\datasets\\Flowers",
    #                 "--dataName", "JunZao",
    #                 "--img_channels", "3",
    #                 "--img_height", "160",
    #                 "--img_width", "160",
    #                 "--batch_size", "8", 
    #                 "--onnx_batch_size", "6",
    #                 "--epochs", "100",
    #                 "--lrepochs", "20,40,60,80:2",
    #                 "--savepthdir", "D:\\MyNetworks\\datasets\\Flowers\\save160\\MobileNetV2\\pth\\",
    #                 "--saveonnxdir", "D:\\MyNetworks\\datasets\\Flowers\\save160\\MobileNetV2\\onnx\\",
    #                 "--savelogdir", "D:\\MyNetworks\\datasets\\Flowers\\save160\\MobileNetV2\\log.txt",
    #                 "--resum"])
    
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.makedirs(args.savepthdir, exist_ok=True)
    os.makedirs(args.saveonnxdir, exist_ok=True)
    logger = get_logger(args.savelogdir)
    
    main()