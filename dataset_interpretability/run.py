from ultralytics import YOLO
from scipy.stats import entropy
import argparse
import time
import numpy as np
from ultralytics.models.yolo.detect.rareplanes_val import RareplanesDetectionValidator
from ultralytics.models.yolo.detect.dimo_val import DIMODetectionValidator
from ultralytics.models.yolo.detect.wasabi_val import WASABIDetectionValidator


# Training should be synthetic images; Validation/Unseen set should be Real images

# Annotation file can be the same. Just two inputs incase they aren't

def get_v_info(train_annotation_file, validation_annotation_file, dataset="rareplanes", image_size=512):
    # Load the YOLO model
    model = YOLO("yolo11n.pt")
    results = model.train(data=train_annotation_file, epochs=10, freeze=10, device=0, imgsz=image_size)
    
    args = dict(model="yolo11n.pt", data=validation_annotation_file, device=0, imgsz=image_size)
    
    if dataset == "rareplanes":
        results2 = RareplanesDetectionValidator(args=args)
        results2()
    elif dataset == "dimo":
        results2 = DIMODetectionValidator(args=args)
        results2()
    elif dataset == "wasabi":
        results2 = WASABIDetectionValidator(args=args)
        results2()
    else:
        model2 = YOLO("yolo11n.pt")
        results2 = model2.val(data=validation_annotation_file, device=0, imgsz=image_size)
    
    # Calculate the entropy of the validation set
    conditional_iou = -1 * calculate_entropy(results.iou_stores)
    predictive_iou = -1 * calculate_entropy(results2.iou_stores)
    
    conditional_conf = -1 * calculate_entropy(results.conf_stores)
    predictive_conf = -1 * calculate_entropy(results2.conf_stores)
    
    conditional_fusion = -1 * calculate_entropy(results.iou_stores * results.conf_stores)
    predictive_fusion = -1 * calculate_entropy(results2.iou_stores * results2.conf_stores)
    
    v_info_iou = predictive_iou - conditional_iou
    v_info_conf = predictive_conf - conditional_conf
    v_info_fusion = predictive_fusion - conditional_fusion
    
    return conditional_iou, predictive_iou, v_info_iou, conditional_conf, predictive_conf, v_info_conf, conditional_fusion, predictive_fusion, v_info_fusion
    
    
def calculate_entropy(p_y_given_f_x):
    p_y_given_f_x[p_y_given_f_x == 0] = 10**-5
    entr = entropy(p_y_given_f_x, base=2)
    return entr

def main():
    parser = argparse.ArgumentParser(description="Calculate V-Info metric.")
    parser.add_argument(
        "--train_annotation_file", 
        type=str, 
        required=True, 
        help="Path to the training annotation file."
    )
    parser.add_argument(
        "--validation_annotation_file", 
        type=str, 
        required=True, 
        help="Path to the validation annotation file."
    )

    args = parser.parse_args()
    
    start_time = time.time()
    conditional_iou, predictive_iou, v_info_iou, conditional_conf, predictive_conf, v_info_conf, conditional_fusion, predictive_fusion, v_info_fusion = get_v_info(args.train_annotation_file, args.validation_annotation_file)
    end_time = time.time()
    
    print(f"conditional_iou: {conditional_iou}")
    print(f"predictive_iou: {predictive_iou}")
    print(f"v_info_iou: {v_info_iou}")
    
    print(f"conditional_conf: {conditional_conf}")
    print(f"predictive_conf: {predictive_conf}")
    print(f"v_info_conf: {v_info_conf}")
    
    print(f"conditional_fusion: {conditional_fusion}")
    print(f"predictive_fusion: {predictive_fusion}")
    print(f"v_info_fusion: {v_info_fusion}")
    
    print(f"Execution Time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
    