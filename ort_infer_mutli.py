import os
import numpy as np
import onnxruntime
import cv2 as cv
from torchvision import transforms
import torch
from shutil import copy, rmtree
from tqdm import tqdm
import json
import time
import pickle


def BlobFromNormalizeRGBImage(img):
    """
        将输入的图片处理为网络输入所需求的格式
        return: [N, C, H, W]
    """
    meanlist = [0.5, 0.5, 0.5]
    stdlist = [0.5, 0.5, 0.5]

    img = img / 255
    R, G, B = cv.split(img)
    if meanlist is not None:
        R = R - meanlist[0]
        G = G - meanlist[1]
        B = B - meanlist[2]

    if stdlist is not None:
        R = R / stdlist[0]
        G = G / stdlist[1]
        B = B / stdlist[2]

    merged = cv.merge([R, G, B])

    blob = merged.transpose((2, 0, 1))

    # blob = np.expand_dims(merged, 0)

    return blob


# def softmax(x):
#     x_exp = np.exp(x)
#     #如果是列向量，则axis=0
#     x_sum = np.sum(x_exp, axis = 0, keepdims = True)
#     s = x_exp / x_sum  
#     return s 

def postprocess(result):
    return torch.softmax(np.array(result)).tolist


def infer(root_path, onnx_path, batch_size, image_size, txt_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root_path = root_path
    assert os.path.exists(root_path)

    ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])

    # ort_session = onnxruntime.InferenceSession(onnx_path)
    # 读取json文件,获取分类信息
    # json_label_path = json_path
    # assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    # json_file = open(json_label_path, 'r')
    # class_indict = json.load(json_file)

    img_name_list = os.listdir(root_path)
    file_num = len(img_name_list)
    # 保证图像数量为batch_size的整数倍
    y = file_num % batch_size
    if y != 0:
        new_img_name_list = img_name_list[y:]
    else:
        new_img_name_list = img_name_list[:]

    new_img_name_tuple = tuple(new_img_name_list)

    # 处理图像
    # new_image_list = []
    # for img_name in new_img_name_tuple:
    #     img = cv.imread(os.path.join(root_path, img_name))
    #     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #     img = cv.resize(img, (image_size, image_size))
    #     blob = BlobFromNormalizeRGBImage(img)
    #     new_image_list.append(blob)

    batch = 0
    start_time = time.time()
    predict_dict = {}

    for i in tqdm(range(0, len(new_img_name_tuple), batch_size), ncols=0):

        new_image_list = []
        for img_name in new_img_name_tuple[i: i + batch_size]:
            img = cv.imread(os.path.join(root_path, img_name))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(img, (image_size, image_size))
            blob = BlobFromNormalizeRGBImage(img)
            new_image_list.append(blob)

        # infer = new_image_list[i: i + batch_size]

        infer = new_image_list

        infer_result = ort_session.run([], {'input1': infer})[0]
        for j in range(batch_size):
            # predict = softmax(infer_resoot_path, img_name))
            #             img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            #             img = cv.resize(img, (image_sult[j])
            predict = torch.softmax(torch.from_numpy(infer_result[j]), dim=0)
            classid = np.argmax(predict)  # 最大索引值
            prob = predict[classid]  # 最大的置信度
            # class_name = class_indict[str(classid)]
            # 统计每一类的数量
            if classid.item() not in predict_dict:
                predict_dict[classid.item()] = 1
            else:
                predict_dict[classid.item()] += 1

            # old_img_path = os.path.join(root_path, new_img_name_list[batch * batch_size + j])
            # new_path = os.path.join(save_path, "{}".format(classid))
            # if not os.path.exists(new_path):
            #     os.makedirs(new_path)
            # new_img_path = os.path.join(new_path, '{}_{:.6f}_{}_{}'.format(classid, prob, i, j) + '.bmp')
            # copy(old_img_path, new_img_path)
        batch += 1

    predict_dict = sorted(predict_dict.items(), key=lambda x: x[0])
    print(predict_dict)
    with open(txt_path, "a") as f:
        f.write(onnx_path + "--" + str(predict_dict) + "\n")

    # print(onnx_path, predict_dict)
    # for i in predict_dict:
    #     print(i)

    # end_time = time.time()
    # avg_time = (end_time - start_time) / len(new_img_name_tuple)
    # print("infer avg time: %.3f ms" % (avg_time * 1e3))


# 修改部分

def main():
    # root_path = r"D:\MyNetworks\datasets\back0223\pipiover\guozi"  # 图像路径
    root_path = r"D:\MyNetworks\datasets\JunZao\test\2zzd_pp"  # 图像路径
    models_path = r"D:\MyNetworks\datasets\JunZao\save\MobileNetV2-0224\onnx"
    batch_size = 6
    image_size = 160
    txt_path = r"D:\MyNetworks\datasets\JunZao\save\MobileNetV2-0224\onnxRandomSampler_val_txt_result.txt"
    print("onnx run......")

    all_onnx_path = os.listdir(models_path)
    for path in all_onnx_path:
        onnx_path = os.path.join(models_path, path)
        infer(root_path, onnx_path, batch_size, image_size, txt_path)


if __name__ == "__main__":
    main()
