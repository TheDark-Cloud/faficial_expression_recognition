import shutil
import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt

def run_selective_search(image_path, image=False, max_regions=10, mode='fast', plot=False):
    # load image

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Invalid Image Path\n'
                         f'Could not load image: {image_path}')

    # Initialize Selective Search Segmentation
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    if mode == 'fast': ss.switchToSelectiveSearchFast()
    else:   ss.switchToSelectiveSearchQuality()

    # Run selective search
    rects = ss.process()
    print(f'Total regions proposal: {len(rects)}')
    for i, (x, y, w, h) in enumerate(rects):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Top {max_regions} Regions Proposal')

    return rects[:max_regions]

def region_of_interest(data_path, image_size=(224,224), path='region_proposal', max_region=300):
    """
    Generate region proposals for all images in dataset_path.
    Save cropped and resized ROIs into a new folder 'region/'.
    """

    # Reset region folder
    save_path = os.path.join(os.path.dirname(data_path), path)
    if os.path.exists(save_path):
        print(f"Deleting old region directory: {save_path}")
        shutil.rmtree(save_path)
        print('Directory deleted')
    os.makedirs(save_path)

    # Loop through each class folder
    for cls in os.listdir(data_path):
        print('Starting region proposals for class: ' + cls)
        src_folder = os.path.join(data_path, cls)
        dest_folder = os.path.join(save_path, cls)
        os.makedirs(dest_folder)

        # Loop through each image in the class
        for file in os.listdir(src_folder):
            img_path = os.path.join(src_folder, file)
            if not os.path.isfile(img_path):
                continue

            # Load image with cv2
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            if img is None:
                print(f"Corrupted file: {img_path}")
                os.remove(img_path)
                continue

            # Convert to NumPy for OpenCV
            img_np = img.numpy()
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Apply Selective Search
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            ss.setBaseImage(img_bgr)
            ss.switchToSelectiveSearchQuality()
            rects = ss.process()[:max_region]

            # Crop, resize, and save proposals
            for i, (x, y, w, h) in enumerate(rects):
                roi = img_np[y:y+h, x:x+w]  # Crop region
                if roi.size == 0:
                    continue

                # Resize
                roi_resized = tf.image.resize(roi, image_size)

                # Convert back to NumPy for saving
                roi_save = (roi_resized.numpy() * 255).astype(np.uint8)

                # Build save path
                save_name = f"{os.path.splitext(file)[0]}_prop{i}.jpg"
                save_path_img = os.path.join(dest_folder, save_name)

                # Save proposal image
                cv2.imwrite(save_path_img, cv2.cvtColor(roi_save, cv2.COLOR_RGB2BGR))

    print("Region proposals generated and saved for all classes.")
    return save_path



