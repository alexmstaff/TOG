from dataset_utils.preprocessing import letterbox_image_padded, image_resize
from misc_utils.visualization import visualize_detections
from keras import backend as K
from models.yolov3 import YOLOv3_Darknet53
from models.yolov3 import YOLOv3_single_416
from models.yolov3 import YOLOv3_single_1920
from models.yolov3 import YOLOv3_single_832
from models.yolov3 import YOLOv3_double_832

from PIL import Image
from tog.attacks import *
import os
from glob import glob
from random import sample

K.clear_session()


source_weights = "../models/yolov3-super-hr_final.h5"
target_weights = "../models/yolov3-cocoV2-merged-hr_best.h5"


source = YOLOv3_single_832(weights=source_weights)
target = YOLOv3_single_832(weights=target_weights)

eps = 32 / 255.0  # Hyperparameter: epsilon in L-inf norm - Default: 8
eps_iter = 2 / 255.0  # Hyperparameter: attack learning rate - Default: 2
n_iter = 10  # Hyperparameter: number of attack iterations - Default: 10


def getImagesInDir(dir_path="data/test"):
    image_list = []
    files = (
        glob(dir_path + "/*.jpeg")
        + glob(dir_path + "/*.jpg")
        + glob(dir_path + "/*.png")
    )
    for filename in files:
        image_list.append(filename)

    return image_list


image_paths = getImagesInDir("/cluster/work/alexamst/yolo-train/test")

selected_im = sample(range(0, 780), 10)

def pre_pro_image(image):
    if len(image.shape) == 4:
        image = image[0]
    im = image * 255
    im = Image.fromarray((im).astype(np.uint8))
    return im


def generate_sample(im, im_num):
    im_name = im.split('/')[-1]
    input_img = Image.open(im)
    # x_query, x_meta = letterbox_image_padded(input_img, size=source.model_img_size)
    x_query = image_resize(input_img, source.model_img_size)
    detections_source_query = source.detect(
        x_query, conf_threshold=source.confidence_thresh_default
    )

    clean_out = pre_pro_image(x_query)
    clean_out.save(os.path.join(os.getcwd(), "output", "clean", im_name))

    detections_target_query = target.detect(
        x_query, conf_threshold=target.confidence_thresh_default
    )

    # Generation of the adversarial example
    x_adv_untargeted = tog_untargeted(
        victim=source, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter
    )

    ad_im = pre_pro_image(x_adv_untargeted)
    ad_im.save(
        os.path.join(os.getcwd(), "output", "untarget", im_name)
    )
    # ad_cropped_im.save(
    #     os.path.join(os.getcwd(), "output", str(im_num) + "_" + "cropped.jpg")
    # )

    # Visualizing the detection results on the adversarial example and compare them with that on the benign input
    detections_adv_untargeted = source.detect(
        x_adv_untargeted, conf_threshold=source.confidence_thresh_default
    )
    visualize_detections(
        {
            "Benign (No Attack)": (
                x_query,
                detections_source_query,
                source.model_img_size,
                source.classes,
            ),
            # "TOG-untargeted": (
            #     x_adv_untargeted,
            #     detections_adv_untargeted,
            #     source.model_img_size,
            #     source.classes,
            # ),
        },
        "visu/" + str(im_num) + "_" + "untarget.png",
    )

    detections_trans_untargeted = target.detect(
        x_adv_untargeted, conf_threshold=target.confidence_thresh_default
    )
    visualize_detections(
        {
            "Benign (No Attack)": (
                x_query,
                detections_target_query,
                target.model_img_size,
                target.classes,
            ),
            "TOG-untargeted": (
                x_adv_untargeted,
                detections_trans_untargeted,
                target.model_img_size,
                target.classes,
            ),
        },
        "visu/" + str(im_num) + "_" + "trans_untarget.png",
    )

    # Generation of the adversarial example
    x_adv_vanishing = tog_vanishing(
        victim=source, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter
    )

    ad_im = pre_pro_image(x_adv_vanishing)
    ad_im.save(os.path.join(os.getcwd(), "output", "vanish", im_name))

    # Visualizing the detection results on the adversarial example and compare them with that on the benign input
    detections_adv_vanishing = source.detect(
        x_adv_vanishing, conf_threshold=source.confidence_thresh_default
    )
    visualize_detections(
        {
            "Benign (No Attack)": (
                x_query,
                detections_source_query,
                source.model_img_size,
                source.classes,
            ),
            "TOG-vanishing": (
                x_adv_vanishing,
                detections_adv_vanishing,
                source.model_img_size,
                source.classes,
            ),
        },
        "visu/" + str(im_num) + "_" + "vanish.png",
    )

    # Generation of the adversarial example
    # x_adv_fabrication = tog_fabrication(
    #     victim=source, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter
    # )

    # ad_im = pre_pro_image(x_adv_fabrication)
    # ad_im.save(os.path.join(os.getcwd(), "output",
    #            str(im_num) + "_" + "fabrication.jpg"))

    # # Visualizing the detection results on the adversarial example and compare them with that on the benign input
    # detections_adv_fabrication = source.detect(
    #     x_adv_fabrication, conf_threshold=source.confidence_thresh_default
    # )
    # visualize_detections(
    #     {
    #         "Benign (No Attack)": (
    #             x_query,
    #             detections_source_query,
    #             source.model_img_size,
    #             source.classes,
    #         ),-*
    #         "TOG-fabrication": (
    #             x_adv_fabrication,
    #             detections_adv_fabrication,
    #             source.model_img_size,
    #             source.classes,
    #         ),
    #     },
    #     str(im_num) + "_" + "fabrication.png",
    # )


# for im in selected_im:
#     generate_sample(image_paths[im], im)
for im in image_paths:
    if im.split('/')[-1] == "bilderEp10_scene103801.jpg":
        generate_sample(im, 0)