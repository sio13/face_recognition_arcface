# Face Recognition using ArcFace


## Train model

```bash
    python3 train.py [-h] [--batch_size BATCH_SIZE] [--max_epochs MAX_EPOCHS] [--step_size STEP_SIZE] [--num_classes NUM_CLASSES] [--num_features NUM_FEATURES]
```

Without passing specific dataset the training will be performed using random data.

## Run a basic demo

```bash
    cd demo/
    python3 face_recognition_camera_cv.py [-h] [--method METHOD] [--tolerance TOLERANCE]
```

Add custom images to the folder `demo/images/` and to the program. Follow the pattern in the script.