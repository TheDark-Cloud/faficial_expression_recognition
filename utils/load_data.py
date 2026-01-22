from tensorflow.keras.utils import image_dataset_from_directory


def load_data(PROCESSED_DATA, BATCH_SIZE, IMG_SIZE, SEED):
    train_ds = image_dataset_from_directory(
        PROCESSED_DATA,
        labels="inferred",  # infer labels from folder names
        label_mode="int",  # labels as integers (0–7)
        image_size=IMG_SIZE,  # resize images
        color_mode='rgb',
        batch_size=BATCH_SIZE,  # batches for training
        shuffle=True,  # shuffle dataset
        validation_split=0.2,  # reserve 20% for validation
        subset="training",  # training split
        seed=SEED  # reproducibility
    )
    val_ds = image_dataset_from_directory(
        PROCESSED_DATA,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=.2,
        subset='validation',
        seed=SEED,
        labels="inferred",
        color_mode="rgb",
        label_mode="int",
    )
    test_ds = image_dataset_from_directory(
        PROCESSED_DATA,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        labels="inferred",
        color_mode="rgb",
        label_mode="int",
    )

    return train_ds, val_ds, test_ds
