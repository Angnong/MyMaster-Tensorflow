import os
import sys

# we put all lfw variants under same folder
# LFW_path = '/media/muellerer/Data/Databases/faceRecognition/images/LFW'
LFW_path = '/home/zhanga/master/database/LFW'
LFW_prefixes = ['lfw','lfw_mb', 'lfw_n']
# LFW_prefix_img1 = 'lfw'
# LFW_prefix_img2 = 'lfw'
model_path = '/home/zhanga/master/TensorFlow/CNN_Training/savemodel/20170313_CASIA-WebFace_resnet_v1_152_rgb_96x96_rcnn'
models_to_test = ['model.ckpt-40000', 'model.ckpt-50000', 'model.ckpt-60000']

#models_to_test = ['model.ckpt-180001', 'model.ckpt-185001', 'model.ckpt-190001', 'model.ckpt-195001']


def main():
    for model in models_to_test:
        cont_prefix1_pos = 0
        cont_prefix2_pos = 0
        for LFW_prefix_img1 in LFW_prefixes:
            for LFW_prefix_img2 in LFW_prefixes:
                print('python3 get_lfw_results.py -l ' + LFW_path + ' -p1 ' + LFW_prefix_img1 + ' -p2 ' + LFW_prefix_img2 +
                ' -m ' + os.path.join(model_path, model))
                os.system('python3 get_lfw_results.py -l ' + LFW_path + ' -p1 ' + LFW_prefix_img1 + ' -p2 ' + LFW_prefix_img2 + ' -m ' + os.path.join(model_path, model))

if __name__ == '__main__':
    main()
