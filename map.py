import os
import subprocess
import sys

#Calculating mAP for.*(\n.*){3}|class_id.*(\n.*){7}

def generate_test(dir):
    image_files = []
    os.chdir(dir)
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".jpg"):
            image_files.append(os.getcwd() + "/" + filename)
    os.chdir("../..")
    with open("test.txt", "w") as outfile:
        for image in image_files:
            outfile.write(image)
            outfile.write("\n")
        outfile.close()


model_types = {
    "single-hr": ("yolov3-single-hr.cfg", "single.data"),
    "double-hr": ("yolov3-double-hr.cfg", "double.data"),
    "double": ("yolov3-double.cfg", "double.data"),
}

image_types = ["clean", "untarget", "vanish"]
model_base_path = os.path.join("/", "cluster", "work", "alexamst", "tog", "models")
models = {
    # "yolov3-bb_final.weights": "double",
    "yolov3-bb-hr_final.weights": "double-hr",
    "yolov3-coco-merged-hr_final.weights": "single-hr",
    "yolov3-smd-merged-hr_final.weights": "single-hr",
    # "yolov3-smd-hr_final.weights": "single-hr",
    "yolov3-boat-hr_final.weights": "single-hr",
    "yolov3-super-hr_final.weights": "single-hr",
    "yolov3-cocoV2-merged-hr_best.weights": "single-hr"
}

# os.chdir(os.path.join("/", "cluster", "home", "alexamst", "TOG", "output"))
os.chdir(os.path.join("/", "cluster", "work", "alexamst", "tog"))

image_types = ["clean/yolov3-super-hr_final-416", "untarget/yolov3-super-hr_final-416", "vanish/yolov3-super-hr_final-416"]

f = open("map.txt", "w")
sys.stdout = f
for type in image_types:
    generate_test(type)
    for model in models:
        print(flush=True)
        print("Calculating mAP for " + type + " images.", flush=True)
        print("yolov3-bb-hr" + " is the source model", flush=True)
        print(model + " is the target model", flush=True)
        print(flush=True)
        subprocess.run(
            [
                "/cluster/home/alexamst/darknet/darknet",
                "detector",
                "map",
                "/cluster/work/alexamst/tog/" + model_types[models[model]][1],
                model_base_path + "/" + model_types[models[model]][0],
                model_base_path + "/" + model,
            ], stdout=f
        )

f.close()
