import os
import sys

model_path = '/home/zhanga/master/TensorFlow/CNN_Training/savemodel/20170315_CASIA-WebFace_resnet_v1_101_rgb_96x96_rcnn/model.ckpt-90000'
img_data_path = '/home/zhanga/master/database/CASIA/CASIA-WebFace'
img_list_path = '/home/zhanga/master/database/CASIA/tf_records/only_ori_try.csv'
bbox_path = '/home/zhanga/master/database/CASIA/faces_all_0.95'
var_list_path = '/home/zhanga/master/database/CASIA/tf_records/only_var_try.csv'


def main():
    os.system('python3 DR_Compair.py -p ' + img_list_path + ' -bbox ' + bbox_path + ' -m ' + model_path + ' -i ' +
              img_data_path + ' -var ' + var_list_path)


if __name__ == '__main__':
    main()
