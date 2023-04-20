import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import swin_tiny_patch4_window7_224 as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create model
    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = "./weights/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # # read class_indict
    # json_path = './class_indices.json'
    # assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    #
    # with open(json_path, "r") as f:
    #     class_indict = json.load(f)

    class_indict = {"0": 0,
                    "1": 1}

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    # load image
    # img_path = "../tulip.jpg"

    s = ""
    num_00 = 0
    num_01 = 0
    num_10 = 0
    num_11 = 0

    num = 0

    src_dir = r"../../partial_imgs/test"
    for src_dir_1 in os.listdir(src_dir):
        src_dir_1_path = os.path.join(src_dir, src_dir_1)
        for file_name in os.listdir(src_dir_1_path):
            num += 1
            img_path = os.path.join(src_dir_1_path, file_name)

            # img_path = r"E:\LJW\Git\deep-learning-for-image-processing\partial_imgs\test\0\04_1_032_head_no_0_0__1_1_1.JPG"
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path)
            plt.imshow(img)
            # [N, C, H, W]
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

                s = s + "{} {} {}\n".format(img_path, src_dir_1, predict_cla)

                if str(src_dir_1) == "0" and str(predict_cla) == "0":
                    num_00 += 1
                elif str(src_dir_1) == "0" and str(predict_cla) == "1":
                    num_01 += 1
                elif str(src_dir_1) == "1" and str(predict_cla) == "0":
                    num_10 += 1
                elif str(src_dir_1) == "1" and str(predict_cla) == "1":
                    num_11 += 1
                print(num, img_path, src_dir_1, predict_cla)
                #
                # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                #                                              predict[predict_cla].numpy())
                # plt.title(print_res)
                # for i in range(len(predict)):
                #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                #                                               predict[i].numpy()))
                # plt.show()


    save_path = "results.txt"
    f = open(save_path, "w", encoding="utf-8")
    f.write(s)
    f.close()
    s2 = "00:{}\n01:{}\n10:{}\n11:{}".format(num_00, num_01, num_10, num_11)

    save_path = "summary_results.txt"
    f = open(save_path, "w", encoding="utf-8")
    f.write(s2)
    f.close()

if __name__ == '__main__':
    main()

