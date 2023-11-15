import os
import sys
import torch
from glob import glob

HOME = os.getcwd()
print("HOME:", HOME)

# %cd {HOME}
# !git clone https://github.com/IDEA-Research/GroundingDINO.git  # https://github.com/facebookresearch/dinov2.git
# %cd {HOME}/GroundingDINO
# !git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
# !pip install -q -e .

# # %cd {HOME}
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swinb_cogcoor.pth")
print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))
 
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py")

print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))

# %cd {HOME}
# !mkdir -p {HOME}/weights
# %cd {HOME}/weights

# !wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
# !wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth


SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE:", DEVICE)


from groundingdino.util.inference import Model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)


SAM_ENCODER_VERSION = "vit_h"
'''
The SAM model can be loaded with 3 different encoders: ViT-B, ViT-L, and ViT-H.
ViT-H improves substantially over ViT-B but has only marginal gains over ViT-L.
These encoders have different parameter counts, with ViT-B having 91M,
ViT-L having 308M, and ViT-H having 636M parameters.
'''


from segment_anything import sam_model_registry, SamPredictor

print("Loading models...")
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)




from typing import List

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}"
        for class_name
        in class_names
    ]

enhance_class_name(class_names=CLASSES)


import cv2
import supervision as sv
import pickle
import gzip
import bz2
import numpy as np
from segment_anything import SamPredictor
import time
import datetime



def save_detection(detection, save_dir=""):
    image_fname = detection[0]
    detection_fname = image_fname[:-4] + '.pkl.bz2'
    if os.path.exists(save_dir):
        basename = os.path.basename(detection_fname)
        detection_fname = os.path.join(save_dir, basename)
    # saving object to disk
    with bz2.open(detection_fname, 'wb') as file:
        # print(detection_fname)
        pickle.dump(detection, file)

def add_class_name_to_detection(detections, class_names):

    try:
        detected_ids = list(detections.class_id)
        detected_names = []
        if len(detected_ids) == 0:
            detected_names = []
        else:
            for i in detected_ids:
                try:
                    name = class_names[i]
                except:
                    name = ""
                detected_names.append(name)

        detections.class_names = detected_names
        # print(detections.class_names)

    except Exception as e:
        print("Error in add_class_name_to_detection():", e, detections)

    return detections

def get_mask_from_detection(image, detections):
    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

def show_detection_segment(image, detections):
# annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    # mask_annotator = sv.MaskAnnotator(color=sv.Color.green())
    
    
    try:
        get_mask_from_detection(image, detections)
        # labels = [
        #     f"{CLASSES[class_id]} {confidence:0.2f}"
        #     for _, _, confidence, class_id, _
        #     in detections]  # Has a bug: DINO may return NoneType class_id,
        # then invoke "list indices must be integers or slices, not NoneType"

        labels = detections.class_names
        # print(labels)

        # draw mask
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

        # draw box
        # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        return annotated_image
    except Exception as e:
        print("Error in show_detection_segment():", e)
        return image

    # %matplotlib inline
    # sv.plot_image(annotated_image, (16, 16))



def dino_detection(image_cv):
    try:
         # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image_cv,
            classes=enhance_class_name(class_names=CLASSES),
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        # print(detections)
        detections = add_class_name_to_detection(detections, CLASSES)

    except Exception as e:
        print("Error in dino_detection:", e)

    return detections


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def detection_segment(img_file_list, seg=True, save_img=True, save_dir=""):
    img_detect_results = []
    
    # Note the start time
    start_time = time.perf_counter()

    def save_seg():
        annotated_image = show_detection_segment(image=image, detections=detections)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        annotated_img_fname = img_file[:-4] + '_annotated.jpg'
        basename = os.path.basename(annotated_img_fname)
        annotated_img_fname = os.path.join(save_dir, basename)
        print("annotated_img_fname:", annotated_img_fname)
        cv2.imwrite(annotated_img_fname, annotated_image)

    
    for idx, img_file in enumerate(img_file_list):
        detection_fname = img_file[:-4] + '.pkl.bz2'
        if os.path.exists(detection_fname):
            print(f"{detection_fname} exists, skipped!")
            continue
        
        print(f"Processing {idx + 1} / {len(img_file_list)}, {img_file}")
        try:
            image = cv2.imread(img_file)
            # detect objects
            detections = dino_detection(image_cv=image)
            if seg:
                get_mask_from_detection(image, detections)

            img_detect_results.append((img_file, detections))

            save_detection((img_file, detections), save_dir=save_dir)

            if save_img:
                save_seg()

                # show image
                # sv.plot_image(annotated_image, (16, 16))
            if (idx % 1000 == 0) or (idx < 10):
                save_seg()

            elapsed_time = time.perf_counter() - start_time
            avg_time_per_img = elapsed_time / (idx+1)
            remaining_imgs = len(img_file_list) - (idx+1)
            remaining_time = avg_time_per_img * remaining_imgs
            
            print(f"Elapsed time: {delta_time(elapsed_time)}. Estimated remaining time: {delta_time(remaining_time):}.")

            processed_cnt += 1
 
        except Exception as e:
            print("Error in detection_segment():", idx, img_file, e)
            continue
    return img_detect_results

# img_file_list = ['/content/drive/MyDrive/Research/street_mapping/thumbnails/LRmIjwUOwYZEOH45ynWNRg_0.0_0.0_0_230.58_L90.jpg']
# img_file_list = ['/content/drive/MyDrive/Research/street_mapping/thumbnails/CU3-cZNDEXJc3YN6IeY3AQ_0.0_0.0_0_297.81_L135.jpg']

CLASSES = ['tree trunk']   # DO NOT use "tree_trunk", will return NoneType for class_id
# CLASSES = ['tree_trunk', 'vehicle',  'person', 'sidewalk', 'road', 'building', 'grass', 'bare earth']   # , 'tree root collar'
# CLASSES = ['tree_trunk', 'vehicle',  'person', 'sidewalk', 'grass', 'bare earth']   # , 'tree root collar'


 # DO NOT use "tree-trunk", may return two classes: tree, trunk

BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.25

from typing import List

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}"
        for class_name
        in class_names
    ]
enhance_class_name(class_names=CLASSES)

def delta_time(seconds):
    delta1 = datetime.timedelta(seconds=seconds)
    str_delta1 = str(delta1)
    decimal_digi = 0
    point_pos = str_delta1.rfind(".")
    str_delta1 = str_delta1[:point_pos]
    return str_delta1
    



save_dir = r'/media/huan/HD16T/Research/street_image_mapping/State_college/Thumbnails_fov90_2_trunk'
# img_detect_results = detection_segment(img_file_list[1000:1001], save_dir=save_dir)
# img_detect_results = detection_segment(img_file_list[40000:40001], save_dir=save_dir)
# img_detect_results = detection_segment(img_file_list[1000000:1000001], save_dir=save_dir)

print("Creat image list...")
 
# img_file_list = glob(os.path.join('/content/drive/MyDrive/Research/street_mapping/thumbnals_state_college', '*.jpg'))
img_file_list = glob(os.path.join('/media/huan/HD16T/Research/street_image_mapping/State_college/Thumbnails_fov90_2', '*.jpg'))

img_file_list = img_file_list[::-1]

print("Found image count:", len(img_file_list))
CLASSES = ['tree trunk']   # DO NOT use "tree_trunk", will return NoneType for class_id
 # DO NOT use "tree-trunk", may return two classes: tree, trunk

print("Started to detect...")

BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.1

processed_cnt = 0

img_detect_results = detection_segment(img_file_list[::-1], save_img=False, save_dir=save_dir)


# img_detect_results[-1][1].class_names




























