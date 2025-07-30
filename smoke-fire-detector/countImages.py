#script to simply count the number of images in train, test, and val sets

import os

DIRS = {
    'train': 'smoke-fire/data/train/images',
    'val':   'smoke-fire/data/val/images',
    'test':  'smoke-fire/data/test/images',
}

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}

def count_images(path):
    try:
        return sum(
            1 for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
    except FileNotFoundError:
        print(f"⚠️  Directory not found: {path}")
        return 0

def main():
    for split, folder in DIRS.items():
        cnt = count_images(folder)
        print(f"{split.capitalize():5s}: {cnt} images")

if __name__ == '__main__':
    main()