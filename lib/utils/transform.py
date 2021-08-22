from lib.utils.tools import draw_bbox, crop_rect, sorted_boxes, get_rotate_crop_image
from PIL import Image
import numpy as np
import cv2
import copy
def crnnRecWithBox(im, boxes_list):
    """
    crnn模型，ocr识别
    @@model,
    @@converter,
    @@im:Array
    @@text_recs:text box
    @@ifIm:是否输出box对应的img

    """
    results = []
    boxes_list = sorted_boxes(np.array(boxes_list))
    # converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    # recognition(config, img, model, converter, device)
    for index, box in enumerate(boxes_list):
        # rect = cv2.minAreaRect(box)
        # degree, w, h, cx, cy = rect
        # box = sorted_boxes(box)
        tmp_box = copy.deepcopy(box)
        partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))

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

        partImg = Image.fromarray(partImg_array).convert("RGB")
        #
        # partImg.save("./debug_im/{}.jpg".format(index))

        partImg_ = partImg.convert('L')
        newW, newH = partImg.size
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
    return partImg_,newW,newH
