import cv2
import albumentations as album
from albumentations.pytorch import ToTensorV2
frame_width = 224
frame_height = 224

def apply_augmentation(bbox, class_name, frame):
    """
    Apply data augmentation techniques to an image with a specified bounding box.

    Args:
    - bbox (tuple): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).
    - class_name (str): Class label associated with the bounding box.
    - frame (numpy.ndarray): Original image frame where the augmentation is applied.

    Returns:
    - augmented_data (list): List of tuples (scaled_bbox, class_name, augmented_frame) containing:
      - scaled_bbox (tuple): Adjusted bounding box coordinates after augmentation.
      - class_name (str): The class label associated with the augmented frame.
      - augmented_frame (numpy.ndarray): Image frame after applying the augmentation.
    """

    # Load image
    image = cv2.resize(frame, (frame_width, frame_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define augmentations
    transforms = [
        album.Compose([
            album.HorizontalFlip(p=1),
            ToTensorV2()
        ], bbox_params=album.BboxParams(format='pascal_voc', label_fields=['class_name'])),
        album.Compose([
            album.Blur(blur_limit=3, p=0.2),
            ToTensorV2()
        ], bbox_params=album.BboxParams(format='pascal_voc', label_fields=['class_name'])),
        album.Compose([
            album.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=1),
            ToTensorV2()
        ], bbox_params=album.BboxParams(format='pascal_voc', label_fields=['class_name']))
    ]

    augmented_data = []

    # Apply augmentations
    for transform in transforms[:3]:  # Limit to num_augmentations
        augmented = transform(image=image, bboxes=[bbox], class_name=[class_name])
        augmented_image = augmented['image']
        augmented_bbox = augmented['bboxes'][0]

        # Convert augmented image back to numpy array
        augmented_image_np = augmented_image.permute(1, 2, 0).cpu().numpy()

        # Prepare output in the desired format
        augmented_frame = cv2.cvtColor(augmented_image_np, cv2.COLOR_RGB2BGR)
        scaled_bbox = (int(augmented_bbox[0]), int(augmented_bbox[1]), int(augmented_bbox[2]), int(augmented_bbox[3]))
        augmented_data.append((scaled_bbox, class_name, augmented_frame))

    return augmented_data
