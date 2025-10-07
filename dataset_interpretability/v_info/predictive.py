import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_conditional(annotations_path):
    
    # Define the path to your annotations and images
    #annotations_path = '/bigdata/NFTI/datasets/RarePlanes/real_plus_synthetic/datasets/synthetic_yolo/labels/eval'
    #annotations_path = '/bigdata/NFTI/datasets/RarePlanes/real_plus_synthetic/datasets/real_yolo/labels/eval'
    #images_path = '/bigdata/NFTI/datasets/RarePlanes/real_plus_synthetic/datasets/synthetic_yolo/images/train'

    # Load annotations
    def load_annotations(file_path):
        with open(file_path, 'r') as f:
            annotations = f.readlines()
        return [list(map(float, line.strip().split())) for line in annotations]

    # Initialize a blank heatmap
    image_shape = (512, 512)  # Set the shape to match your images
    heatmap = np.zeros(image_shape, dtype=np.float32)

    # Iterate over annotation files
    for annotation_file in tqdm(os.listdir(annotations_path)):
        if annotation_file.endswith('.txt'):
            annotations = load_annotations(os.path.join(annotations_path, annotation_file))
            
            # Load corresponding image to get dimensions
            #image_file = os.path.join(images_path, annotation_file.replace('.txt', '.png'))
            #image = cv2.imread(image_file)
            #img_height, img_width = image.shape[:2]

            img_height, img_width = image_shape

            # Update heatmap with bounding boxes
            for ann in annotations:
                _, cx, cy, w, h = ann
                x_min = int((cx - w / 2) * img_width)
                x_max = int((cx + w / 2) * img_width)
                y_min = int((cy - h / 2) * img_height)
                y_max = int((cy + h / 2) * img_height)
                
                heatmap[y_min:y_max, x_min:x_max] += 1

    # Normalize heatmap to range [0, 1]
    heatmap /= np.max(heatmap)

    # Plot heatmap
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Bounding Box Heatmap')
    #plt.show()
    plt.savefig("synthetic_eval_heatmap.png")
    
   