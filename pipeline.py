from dataset_utils.preprocessing import letterbox_image_padded, image_resize
from keras import backend as K

# from models.yolov3 import YOLOv3_Darknet53
# from models.yolov3 import YOLOv3_single_1920
# from models.yolov3 import YOLOv3_double_1920
from models.yolov3 import YOLOv3_single_832
from models.yolov3 import YOLOv3_double_832
from models.yolov3 import YOLOv3_single_416


from PIL import Image
from tog.attacks import *
import os
from glob import glob
import argparse

eps = 30 / 255.0  # Hyperparameter: epsilon in L-inf norm
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


def open_and_detect(image_path):
    fpath = image_path
    input_img = Image.open(fpath)
    # x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
    x_query = image_resize(input_img, size=detector.model_img_size)
    return x_query


def generate_adversarial_examples(x_query):
    x_adv_untargeted = tog_untargeted(
        victim=detector, x_query=x_query, n_iter=n_iter, eps=eps/255, eps_iter=eps_iter
    )
    # x_adv_vanishing = tog_vanishing(
    #     victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter
    # )
    return x_adv_untargeted#, x_adv_vanishing


def pre_pro_image(image):
    if len(image.shape) == 4:
        image = image[0]
    im = image * 255
    # im = crop_image_only_outside(im)
    im = Image.fromarray((im).astype(np.uint8))
    return im


def generate_samples(source_model):
    source_model, extension = os.path.splitext(source_model)
    output_clean = "/cluster/work/alexamst/tog/clean/" + source_model + "-" + str(eps) + "/"
    output_vanish = "/cluster/work/alexamst/tog/vanish/" + source_model + "-" + str(eps) + "/"
    output_untargeted = "/cluster/work/alexamst/tog/eps_comp/" + source_model + "-" + str(eps) + "/"

    if not os.path.exists(output_clean):
        os.makedirs(output_clean)
    if not os.path.exists(output_vanish):
        os.makedirs(output_vanish)
    if not os.path.exists(output_untargeted):
        os.makedirs(output_untargeted)

    image_paths = getImagesInDir("/cluster/work/alexamst/yolo-train/test")

    loop_counter = 0

    for image_path in image_paths:
        loop_counter += 1
        print(
            str(loop_counter)
            + "/"
            + str(len(image_paths))
            + "\t Source model: "
            + source_model,
            flush=True,
        )
        clean_sample = open_and_detect(image_path)
        path, filename = os.path.split(image_path)

        # clean_out = pre_pro_image(clean_sample)
        # clean_out.save(os.path.join(output_clean, filename))

        untargeted = generate_adversarial_examples(clean_sample)
        if untargeted.any():
            untargeted_out = pre_pro_image(untargeted)
            untargeted_out.save(os.path.join(output_untargeted, filename))
            print("Untargeted adversarial sample " + image_path + " saved", flush=True)
            untargeted = None
        # if vanish.any():
        #     vanish_out = pre_pro_image(vanish)
        #     vanish_out.save(os.path.join(output_vanish, filename))
        #     print("Vanishing adversarial sample " + image_path + " saved", flush=True)
        #     vanish = None


models = [
    # "yolov3-super-hr_final.h5",
    # "yolov3-bb-hr_final.h5",
    "yolov3-boat-hr_final.h5",
    "yolov3-coco-merged-hr_final.h5",
    # "yolov3-smd-merged-hr_final.h5",
    # "yolov3-smd-hr_final.h5",
    # "yolov3-cocoV2-merged-hr_best.h5",
]

ep_list = [8, 12, 16, 20, 24, 28, 32]

parser = argparse.ArgumentParser()
parser.add_argument("model_index", nargs="?", type=int)
args = parser.parse_args()

if args.model_index:
    model = models[args.model_index]
    print("Generating samples for " + model, flush=True)

    K.clear_session()

    weights = "../models/" + model
    if model == "yolov3-bb-hr_final.h5":
        # K.clear_session()
        # detector = YOLOv3_double_1920(weights=weights)
        # generate_samples("lossless-" + model)
        K.clear_session()
        detector = YOLOv3_double_832(weights=weights)
        generate_samples(model)
    else:
        # K.clear_session()
        # detector = YOLOv3_single_1920(weights=weights)
        # generate_samples("lossless-" + model)
        K.clear_session()
        detector = YOLOv3_single_832(weights=weights)
        generate_samples(model)
    print(model + " finished generating samples", flush=True)

else:
    for linf in ep_list:
        eps = linf
        for model in models:

            K.clear_session()

            weights = "../models/" + model
            if model == "yolov3-bb-hr_final.h5":
                # K.clear_session()
                # detector = YOLOv3_double_1920(weights=weights)
                # generate_samples("lossless-" + model)
                K.clear_session()
                detector = YOLOv3_double_832(weights=weights)
                generate_samples(model)
            else:
                # K.clear_session()
                # detector = YOLOv3_single_1920(weights=weights)
                # generate_samples("lossless-" + model)
                K.clear_session()
                detector = YOLOv3_single_832(weights=weights)
                generate_samples(model)
            print(model + " finished generating samples", flush=True)
