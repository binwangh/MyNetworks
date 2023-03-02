import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from models.model_MobileNetV2 import MobileNetV2

import shutil

def GetAllFilePath(oneDirectoryPath, resultPath):
    f = open(resultPath, 'w')

    primesense_list = []
    oneDirectory = os.walk(oneDirectoryPath)
    for path, dir_list, file_list in oneDirectory:
        for file_name in file_list:
            str_path = os.path.join(path, file_name)
            primesense_list.append(str_path)
            f.write(str_path)
            f.write("\n")

    # print("success!!!")

    return primesense_list


def main():
    
    dataPre = GetAllFilePath(r"D:\\MyNetworks\\datasets\\JunZao\\train", r"D:\\MyNetworks\\datasets\\JunZao\\train.srcData.txt")
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device = ", device)
    data_transform = transforms.Compose(
        [transforms.Resize([160, 160]),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    # read class_indict
    json_path = 'D:\\MyNetworks\\datasets\\JunZao\\class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    # model = MobileNetV2(num_classes=len(class_indict)).to(device)
    model = MobileNetV2(num_classes=len(class_indict))
    # model = torch.nn.DataParallel(model)
    model.to(device)
    # load model weights
    model_weight_path = "D:\\MyNetworks\\datasets\\JunZao\\save\\MobileNetV2-0225\\pth\\90_0.8022_0.181_0.692_0.842_0.857_0.972_0.966_0.921_0.835_0.789.pth"
    state_dict = torch.load(model_weight_path, map_location=device)
    model.load_state_dict(state_dict)

    train_bar = tqdm(dataPre, file=sys.stdout)

    # for num_label in range(train_bar):
    for step, data in enumerate(train_bar):
    
        # print(step)
        # print(data)
        # load image
        img_path = data
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        # plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                    predict[predict_cla].numpy())
        # plt.title(print_res)
        # for i in range(len(predict)):
        #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
        #                                             predict[i].numpy()))
        # plt.show()
        
        # print(num_label)
        # print(img_path)
        filename = img_path.split("\\")[-1]
        # print(filename)
        
        if predict[predict_cla].numpy() >= 0.90 and predict_cla == 1:
            dstDirPath_one = r"D:\\MyNetworks\\datasets\\JunZao\\train_re\\"
            dstDirPath =  dstDirPath_one + class_indict[str(predict_cla)]
            dst = dstDirPath + "\\" + filename
            
            if not os.path.isdir(dstDirPath_one):
                # print('The directory is not present. Creating a new one..')
                os.mkdir(dstDirPath_one)
            
            if not os.path.isdir(dstDirPath):
                # print('The directory is not present. Creating a new one..')
                os.mkdir(dstDirPath)
            
            # shutil.copy(img_path, dst)
            shutil.move(img_path, dst)


if __name__ == '__main__':
    main()
