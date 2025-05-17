import os
from PIL import Image
from tqdm import tqdm

# Settings
base_path = "data/train"
num_augmentations = 2
rotation_range = 90

def get_next_index(class_path):
    files = [f for f in os.listdir(class_path) if f.endswith('.png')]
    indices = [int(os.path.splitext(f)[0]) for f in files if os.path.splitext(f)[0].isdigit()]
    return max(indices, default=-1) + 1

def augment_image(img, degrees):
    return img.rotate(degrees, fillcolor=0)

for class_name in os.listdir(base_path):
    class_path = os.path.join(base_path, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"Processing class '{class_name}'...")
    image_files = []
    for f in os.listdir(class_path):
        if f.endswith(".png"):
            name = os.path.splitext(f)[0]
            if name.isdigit():
                image_files.append(f)

    image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))


    next_index = get_next_index(class_path)

    for img_file in tqdm(image_files, desc=f"Class {class_name}"):
        img_path = os.path.join(class_path, img_file)
        img = Image.open(img_path).convert("L")

        for i in range(num_augmentations):
            angle = -rotation_range + (i * (2 * rotation_range / (num_augmentations - 1)))
            augmented_img = augment_image(img, angle)
            save_path = os.path.join(class_path, f"{next_index}.png")
            augmented_img.save(save_path)
            next_index += 1
