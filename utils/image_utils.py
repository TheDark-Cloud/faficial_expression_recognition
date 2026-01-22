import hashlib
import cv2
import os
import tensorflow as tf
import shutil
from config.config import cfg
from matplotlib import pyplot as plt

VALID_EXTS = (".jpg", ".jpeg", ".png")

def img_processing_cv2(path, image_size=(128, 128)):
    print('Image Processing Started...\n')
    classes = os.listdir(path)
    parent_path = os.path.dirname(path)
    processed_dir = os.path.join(parent_path, 'cleaned')

    # Reset cleaned directory
    if os.path.exists(processed_dir):
        print(f'Deleting old directory... {processed_dir}')
        shutil.rmtree(processed_dir)
        print('Directory deleted\n')
    os.makedirs(processed_dir)

    valid_exts = (".jpg", ".jpeg", ".png")

    for cls in classes:
        folder = os.path.join(path, cls)
        if not os.path.isdir(folder):   # skip non-folders
            continue

        save_dir = os.path.join(processed_dir, cls)
        os.makedirs(save_dir, exist_ok=True)

        seen = set()
        for file in os.listdir(folder):
            if not file.lower().endswith(valid_exts):
                continue

            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue

            # Duplicate check (optional)
            with open(img_path, 'rb') as f:
                h = hashlib.md5(f.read()).hexdigest()
            if h in seen:
                continue
            seen.add(h)

            # Resize
            img = cv2.resize(img, image_size)

            # Save (keep BGR to avoid color shift)
            save_path = os.path.join(save_dir, file)
            cv2.imwrite(save_path, img)

    print('Image Cleaning Completed')
    return processed_dir


def img_processing_tf(path, image_size=(128, 128)):
    if os.path.exists(os.path.join(os.path.dirname(path), 'processed')):
        print(f'Deleting old directory...{os.path.join(os.path.dirname(path), 'processed')}')
        shutil.rmtree(os.path.join(path, 'processed'))
        print('Directory Deleted')

    save_dir = os.path.join(os.path.dirname(path), 'processed')
    os.makedirs(save_dir)


    for cls in os.listdir(path):
        folder = os.path.join(path, cls)
        for file in os.listdir(folder):
            if not file.lower().endswith(VALID_EXTS):
                print(f'Invalid Image Processing Failed... {file}\nDeleting file...\n')
                os.remove(os.path.join(folder, file))
                continue

            img_path = os.path.join(folder, file)
            try:
                # Loading with tensorflow (RGB)
                img_bytes = tf.io.read_file(img_path)
                img = tf.image.decode_image(img_bytes, channels=3)
                # Resizing Data
                img = tf.image.resize(img, image_size)
                # Normalizing Data
                img = tf.cast(img, tf.float32)/255.0
                # Augmenting Data
                img  = tf.image.random_flip_left_right(img)
                img = tf.image.random_flip_up_down(img)
                img = tf.image.rot90(img, k=tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32))
                img = tf.image.random_brightness(img, max_delta=0.2)
                # Converting back to uint8 for saving
                img = tf.cast(img*255, tf.uint8)
                # saving to folder
                save_path = os.path.join(save_dir, file)
                encoded = tf.io.encode_png(img)
                tf.io.write_file(save_path, encoded)
            except Exception as err:
                print(f'Error: {err}')
    print('Image Processing Completed')
    print(f'Path: {save_dir}')
    return save_dir


