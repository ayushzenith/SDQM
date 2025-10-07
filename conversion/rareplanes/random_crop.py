import os
import cv2
import json
import albumentations as A
from tqdm import tqdm
from pycocotools.coco import COCO
import argparse
import random
import numpy as np

# Set the random seed for reproducibility
random.seed(42)
np.random.seed(42)


def random_crop_dataset(input_dir, output_dir, input_annotations_path, output_annotations_path, width=512, height=512, min_visibility=0.2):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load COCO annotations
    coco = COCO(input_annotations_path)
    output_annotations = {
        'images': [],
        'annotations': [],
        'categories': coco.loadCats(coco.getCatIds())
    }

    # Define the augmentation pipeline
    transform = A.Compose([
        A.RandomCrop(width=width, height=height, p=1.0),
    ], bbox_params=A.BboxParams(format='coco', min_visibility=min_visibility, label_fields=['category_ids']))

    annotation_id = 1

    # Process each image in the input directory
    for img_id in tqdm(coco.getImgIds()):
        img_info = coco.loadImgs(img_id)[0]
        img_name = img_info['file_name']
        img_path = os.path.join(input_dir, img_name)
        
        # Read the image
        image = cv2.imread(img_path)
        
        # Get annotations for the image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        bboxes = [ann['bbox'] for ann in anns]
        category_ids = [ann['category_id'] for ann in anns]
        
        assert len(bboxes) > 0, f"No annotations found for image {img_name}"
        
        # Apply the augmentation
        while True:
            augmented = transform(image=image, bboxes=bboxes, category_ids=category_ids)
            cropped_image = augmented['image']
            cropped_bboxes = augmented['bboxes']

            if len(cropped_bboxes) > 0:
                break
        
        # Define the path for the output image
        output_path = os.path.join(output_dir, img_name)
        
        # Save the cropped image
        cv2.imwrite(output_path, cropped_image)
        
        # Update image info
        new_img_info = {
            'id': img_id,
            'file_name': img_name,
            'width': width,
            'height': height
        }
        output_annotations['images'].append(new_img_info)
        
        # Update annotations for the image
        for bbox, category_id in zip(cropped_bboxes, category_ids):
            x, y, w, h = bbox
            new_bbox = [x, y, w, h]
            
            new_ann = {
                'id': annotation_id,
                'image_id': img_id,
                'category_id': category_id,
                'bbox': new_bbox,
                'area': w * h,
                'iscrowd': 0,
                'segmentation': [],  # Update if segmentation exists
            }
            output_annotations['annotations'].append(new_ann)
            annotation_id += 1

    # Save new annotations to JSON
    with open(output_annotations_path, 'w') as f:
        json.dump(output_annotations, f)

    print("Processing completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing the images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for the cropped images')
    parser.add_argument('--input_annotations', type=str, required=True,
                        help='Path to the input COCO annotations file')
    parser.add_argument('--output_annotations', type=str, required=True,
                        help='Path to save the output COCO annotations file')
    parser.add_argument('--width', type=int, default=512,
                        help='Width of the cropped images')
    parser.add_argument('--height', type=int, default=512,
                        help='Height of the cropped images')
    parser.add_argument('--min_visibility', type=float, default=0,
                        help='Minimum visibility of the object to keep')
    args = parser.parse_args()

    random_crop_dataset(args.input_dir, args.output_dir, args.input_annotations, args.output_annotations, args.width, args.height, args.min_visibility)
