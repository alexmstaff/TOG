from dataset_utils.preprocessing import letterbox_image_padded, image_resize
from misc_utils.visualization import visualize_detections
from keras import backend as K
from models.yolov3 import YOLOv3_Darknet53
from models.yolov3 import YOLOv3_single_1920
from models.yolov3 import YOLOv3_single_832

from PIL import Image
from tog.attacks import *
import os
from glob import glob
from random import sample

K.clear_session()


weights = "../models/yolov3-super-hr_final.h5"

detector = YOLOv3_single_1920(weights=weights)

eps = 8 / 255.0  # Hyperparameter: epsilon in L-inf norm
eps_iter = 2 / 255.0  # Hyperparameter: attack learning rate
n_iter = 10  # Hyperparameter: number of attack iterations


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

selected_im = sample(range(0, 121), 10)

fpath = image_paths[0]


def crop_image_only_outside(img, tol=80):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img > tol
    if img.ndim == 3:
        mask = mask.all(2)
    m, n = mask.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def pre_pro_image(image):
    if len(image.shape) == 4:
        image = image[0]
    im = image * 255
    cropped_im = crop_image_only_outside(im)
    im = Image.fromarray((im).astype(np.uint8))
    cropped_im = Image.fromarray((cropped_im).astype(np.uint8))
    return im, cropped_im


def generate_sample(im, im_num):
    input_img = Image.open(im)
    # x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
    x_query = image_resize(input_img, detector.model_img_size)
    detections_query = detector.detect(
        x_query, conf_threshold=detector.confidence_thresh_default
    )
    visualize_detections(
        {
            "Benign (No Attack)": (
                x_query,
                detections_query,
                detector.model_img_size,
                detector.classes,
            )
        }
    )

    # Generation of the adversarial example
    x_adv_untargeted = tog_untargeted(
        victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter
    )

    ad_im, ad_cropped_im = pre_pro_image(x_adv_untargeted)
    ad_im.save(
        os.path.join(os.getcwd(), "output", str(im_num) + "_" + "untargeted.jpg")
    )
    # ad_cropped_im.save(
    #     os.path.join(os.getcwd(), "output", str(im_num) + "_" + "cropped.jpg")
    # )

    # Visualizing the detection results on the adversarial example and compare them with that on the benign input
    detections_adv_untargeted = detector.detect(
        x_adv_untargeted, conf_threshold=detector.confidence_thresh_default
    )
    visualize_detections(
        {
            "Benign (No Attack)": (
                x_query,
                detections_query,
                detector.model_img_size,
                detector.classes,
            ),
            "TOG-untargeted": (
                x_adv_untargeted,
                detections_adv_untargeted,
                detector.model_img_size,
                detector.classes,
            ),
        },
        str(im_num) + "_" + "untarget.png",
    )

    # Generation of the adversarial example
    x_adv_vanishing = tog_vanishing(
        victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter
    )

    ad_im, ad_cropped_im = pre_pro_image(x_adv_vanishing)
    ad_im.save(os.path.join(os.getcwd(), "output", str(im_num) + "_" + "vanishing.jpg"))

    # Visualizing the detection results on the adversarial example and compare them with that on the benign input
    detections_adv_vanishing = detector.detect(
        x_adv_vanishing, conf_threshold=detector.confidence_thresh_default
    )
    visualize_detections(
        {
            "Benign (No Attack)": (
                x_query,
                detections_query,
                detector.model_img_size,
                detector.classes,
            ),
            "TOG-vanishing": (
                x_adv_vanishing,
                detections_adv_vanishing,
                detector.model_img_size,
                detector.classes,
            ),
        },
        str(im_num) + "_" + "vanish.png",
    )


for im in selected_im:
    generate_sample(image_paths[im], im)
