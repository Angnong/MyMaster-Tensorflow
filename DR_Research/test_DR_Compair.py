import os
import sys

model_path = '/home/zhanga/master/TensorFlow/CNN_Training/savemodel/20170316_CASIA-WebFace_resnet_v1_50_rgb_96x96_rcnn/model.ckpt-90000'
img_data_path = '/home/zhanga/master/database/CASIA/CASIA-WebFace'
img_list_path = '/home/zhanga/master/database/CASIA/tf_records/20170313_CASIA-WebFace_rcnn_0.9/trainlist_casia_eric.csv'
bbox_path = '/home/zhanga/master/database/CASIA/faces_WebFace_0.9'
var_list_path = 's'


def main():
    os.system('python3 DR_Compair.py -p ' + img_list_path + ' -bbox ' + bbox_path + ' -m ' + model_path + ' -i ' +
              img_data_path + ' -var ' + var_list_path)


if __name__ == '__main__':
    main()
