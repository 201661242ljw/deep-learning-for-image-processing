import json
import os

import cv2
import numpy as np

have_list = os.listdir(r"E:\LJW\Git\deep-learning-for-image-processing\partial_imgs\test\1")
have_no_list = os.listdir(r"E:\LJW\Git\deep-learning-for-image-processing\partial_imgs\test\0")

area = json.load(open(r'E:\LJW\mmpose\00_LJW\area.json', 'r', encoding='utf-8'), strict=False)
area_points_list = []
for tower_mode in area.keys():
    areas = area[tower_mode]
    for a in areas:
        for point_name in a:
            area_points_list.append(point_name)

length = 128
half_length = 64

src_json_path = r"E:\LJW\mmpose\00_LJW\tower_dataset_12456_train_val_test\annotations\0_keypoints_test.json"
dst_json_path = r"E:\LJW\mmpose\00_LJW\resized_dataset\768\annotations\0_keypoints_test.json"
predict_json_path = r"E:\LJW\Git\my_mmpose\mmpose\00_LJW\results\swin_b_p4_w12_768_batch2\result_keypoints.json"

src_data = json.load(open(src_json_path, 'r', encoding='utf-8'), strict=False)
dst_data = json.load(open(dst_json_path, 'r', encoding='utf-8'), strict=False)
predict_data = json.load(open(predict_json_path, 'r', encoding='utf-8'), strict=False)

src_annotations = src_data['annotations']
src_images = src_data['images']
dst_annotations = dst_data['annotations']
dst_images = dst_data['images']

total_num = len(src_images)

point_names = src_data['categories'][0]['keypoint']

for i in range(total_num):
    src_annotation = src_annotations[i]
    src_image = src_images[i]
    dst_annotation = dst_annotations[i]
    dst_image = dst_images[i]

    predict = predict_data[i]

    src_image_id = src_annotation['image_id']
    dst_image_id = dst_annotation['image_id']
    src_id = src_image['id']
    dst_id = dst_image['id']
    predict_image_id = predict['image_id']

    lst = [src_id, src_image_id, dst_id, dst_image_id, predict_image_id]

    assert max(lst) == min(lst)

    dst_points = np.array(dst_annotation['keypoints']).reshape((-1, 3))
    src_points = np.array(src_annotation['keypoints']).reshape((-1, 3))
    predict_points = np.array(predict['keypoints']).reshape((-1, 3))

    src_img_name = src_image['file_name']

    src_img_path = os.path.join(r"E:\LJW\mmpose\00_LJW\tower_dataset_12456_train_val_test\imgs", src_img_name)
    src_img = cv2.imread(src_img_path, 1)

    mask = src_points[:, 2] != 0
    src_points_ = src_points[mask]
    src_x1 = min(src_points_[:, 0])
    src_y1 = min(src_points_[:, 1])
    src_x2 = max(src_points_[:, 0])
    src_y2 = max(src_points_[:, 1])

    mask = dst_points[:, 2] != 0
    dst_points_ = dst_points[mask]
    dst_x1 = min(dst_points_[:, 0])
    dst_y1 = min(dst_points_[:, 1])
    dst_x2 = max(dst_points_[:, 0])
    dst_y2 = max(dst_points_[:, 1])

    h = src_image['height']
    w = src_image['width']
    done_list = []

    data_points = {}

    for j, point_name in enumerate(point_names):
        visible = src_points[j, 2]
        if visible == 0:
            continue

        predict_x = predict_points[j, 0]
        predict_y = predict_points[j, 1]

        x = (predict_x - dst_x1) / (dst_x2 - dst_x1) * (src_x2 - src_x1) + src_x1
        y = (predict_y - dst_y1) / (dst_y2 - dst_y1) * (src_y2 - src_y1) + src_y1
        x = int(x)
        y = int(y)
        data_points[point_name] = [x, y]

    for j, point_name in enumerate(point_names):
        visible = src_points[j, 2]
        if visible == 0:
            continue
        if point_name in area_points_list:
            if point_name not in done_list:
                for tower_mode in area.keys():
                    areas = area[tower_mode]
                    for index, a in enumerate(areas):
                        if point_name in a:
                            # if point_name != '2_5_1':
                            #     continue
                            x1 = 100000
                            y1 = 100000
                            x2 = 0
                            y2 = 0
                            for p_name in a:
                                done_list.append(p_name)
                                x = data_points[p_name][0]
                                y = data_points[p_name][1]
                                x1 = min(x, x1)
                                y1 = min(y, y1)
                                x2 = max(x, x2)
                                y2 = max(y, y2)

                            length = max(half_length * 2, x2 - x1, y2 - y1)
                            x_c = (x1 + x2) // 2
                            y_c = (y1 + y2) // 2
                            x_c = min(max(length // 2, x_c), w - 1 - length // 2)
                            y_c = min(max(length // 2, y_c), h - 1 - length // 2)
                            x1 = x_c - length // 2
                            x2 = x_c + length // 2
                            y1 = y_c - length // 2
                            y2 = y_c + length // 2

                            save_name = src_img_name.split('.')[0] + '__area_{}.JPG'.format(index)
                            if save_name in have_list:
                                have = 1
                            elif save_name in have_no_list:
                                have = 0
                            else:
                                have = 2
                            save_path = os.path.join(
                                r"E:\LJW\Git\deep-learning-for-image-processing\partial_imgs\test_\{}".format(have),
                                save_name)
                            if os.path.exists(save_path):
                                continue
                            img_temp = src_img[y1:y2, x1:x2:]

                            img_temp = cv2.resize(img_temp, (half_length * 2, half_length * 2))
                            print(save_path)
                            cv2.imwrite(save_path, img_temp)

            continue

        x = min(max(half_length, data_points[point_name][0]), w - 1 - half_length)
        y = min(max(half_length, data_points[point_name][1]), h - 1 - half_length)

        save_name = "{}__{}.JPG".format(src_img_name.split(".")[0], point_name)

        if save_name in have_list:
            have = 1
        elif save_name in have_no_list:
            have = 0
        else:
            have = 2

        save_path = os.path.join(
            r"E:\LJW\Git\deep-learning-for-image-processing\partial_imgs\test_\{}".format(have), save_name)
        if os.path.exists(save_path):
            continue
        print(save_path)
        cv2.imwrite(save_path, src_img[y - half_length:y + half_length, x - half_length:x + half_length, :])
