import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
from dbnet.decode import SegDetectorRepresenter
from psenet.PSENET import SingletonType
from dbnet.dbnet_infer import DBNET,draw_bbox
from lib.utils.tools import crop_rect, sorted_boxes, get_rotate_crop_image
from PIL import Image
import numpy as np
import cv2
import copy
from lib.utils.utils import resizeNormalize
def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/360CC_config.yaml')
    parser.add_argument('--image_path', type=str, default='images/test_2.png', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default='output/checkpoints/mixed_second_finetune_acc_97P7.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def recognition(config, img, model, converter, device,h,w):

    # github issues: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec/issues/211
    # h, w = img.shape
    # fisrt step: resize the height and width of image to (32, x)
    # img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)

    # second step: keep the ratio of image's text same with training
    # h, w = img.shape
    # w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    # img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    # img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    # normalize
    # img = img.astype(np.float32)
    # img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    # img = img.transpose([2, 0, 1])
    # img = torch.from_numpy(img)
    #
    # img = img.to(device)
    # img = img.view(1, *img.size())
    model.eval()
    preds = model(img)
    print(preds.shape)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred
    # print('results: {0}'.format(sim_pred))
import matplotlib.pyplot as plt
if __name__ == '__main__':

    config, args = parse_arg()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    started = time.time()
    text_handle = DBNET(MODEL_PATH="./models/dbnet.onnx")
    img = cv2.imread("./images/test3.png")
    print(img.shape)
    box_list, score_list = text_handle.process(img)
    # img = draw_bbox(img, box_list)
    # cv2.imwrite("test.jpg", img)
    boxes_list = sorted_boxes(np.array(box_list))
    # converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    # recognition(config, img, model, converter, device)
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    words=[]
    for index, box in enumerate(boxes_list):
        # rect = cv2.minAreaRect(box)
        # degree, w, h, cx, cy = rect
        # box = sorted_boxes(box)
        tmp_box = copy.deepcopy(box)
        partImg_array = get_rotate_crop_image(img, tmp_box.astype(np.float32))

        # partImg = Image.fromarray(partImg_array).convert("RGB")

        # partImg.save("./debug_im/{}.jpg".format(index))

        # angel_index = angle_handle.predict(partImg_array)
        #
        # angel_class = lable_map_dict[angel_index]
        # # print(angel_class)
        # rotate_angle = rotae_map_dict[angel_class]
        #
        # if rotate_angle != 0:
        #     partImg_array = np.rot90(partImg_array, rotate_angle // 90)
        # recognition(config,)
        partImg = Image.fromarray(partImg_array).convert("RGB")
        #
        # partImg.save("./debug_im/{}.jpg".format(index))

        # partImg_ = partImg.convert('L')
        newW, newH = partImg.size
        im=partImg
        image = im.convert('L')
        scale = image.size[1] * 1.0 / 32
        w = image.size[0] / scale
        w = int(w)
        transformer = resizeNormalize((w, 32))
        plt.imshow(image)

        image = transformer(image)
        image = image.to(device)
        image = image.view(1, *image.size())
        # image = Variable(image)

        plt.show()
        text=recognition(config,image,model,converter,device,newH,newW )
        words.append(text)
        # try:

        # if crnn_vertical_handle is not None and angel_class in ["shudao", "shuzhen"]:
        #
        #     simPred = crnn_vertical_handle.predict(partImg_)
        # else:
        #     simPred = crnn_handle.predict(partImg_)  ##识别的文本
        # except:
        #     continue
        #
        # if simPred.strip() != u'':
        #     results.append({'cx': 0, 'cy': 0, 'text': simPred, 'w': newW, 'h': newH,
        #                     'degree': 0})
        # results.append({ 'text': simPred, })

    # img = cv2.imread(args.image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = draw_bbox(img, box_list,words)
    cv2.imwrite("test.jpg", img)

    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))

