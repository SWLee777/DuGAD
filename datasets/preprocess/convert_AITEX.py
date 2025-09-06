import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import argparse
from tqdm import tqdm
from imghdr import what

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, help="dataset root")
    args = parser.parse_args()

    DEFEAT_CLASS = {
        '002': "Broken_end", '006': "Broken_yarn", '010': "Broken_pick",
        '016': "Weft_curling", '019': "Fuzzyball", '022': "Cut_selvage",
        '023': "Crease", '025': "Warp_ball", '027': "Knots",
        '029': "Contamination", '030': "Nep", '036': "Weft_crack"
    }

    normal_images = list()
    normal_fname = list()
    outlier_images = list()
    outlier_labels = list()
    outlier_fname = list()

    def is_image_file(file_path):
        """检查文件是否为有效的图像文件"""
        try:
            return what(file_path) is not None
        except Exception as e:
            print(f"Error checking image type for {file_path}: {e}")
            return False

    def process_normal_images(root_dir):
        """递归遍历所有子目录，处理正常图像"""
        print(f"Recursively processing normal images from: {root_dir}")
        file_count = 0
        image_count = 0

        # 递归遍历所有目录和文件
        for dirpath, dirnames, filenames in os.walk(root_dir):
            print(f"\nEntering directory: {dirpath}")
            print(f"Found {len(filenames)} files and {len(dirnames)} subdirectories")

            for filename in filenames:
                file_count += 1
                file_path = os.path.join(dirpath, filename)

                # 打印文件基本信息（调试用）
                print(f"\nChecking file {file_count}: {file_path}")
                print(f"Is file? {os.path.isfile(file_path)}")
                print(f"File size: {os.path.getsize(file_path)} bytes" if os.path.exists(file_path) else "File does not exist")

                # 检查是否为有效文件
                if not os.path.isfile(file_path):
                    print(f"Skipping - not a valid file")
                    continue

                # 检查是否为图像文件
                if not is_image_file(file_path):
                    print(f"Skipping - not an image file")
                    continue

                # 尝试读取图像
                image_data = cv2.imread(file_path)
                if image_data is None:
                    print(f"Skipping - failed to read image (corrupted?)")
                    continue

                # 检查图像尺寸
                img_height, img_width = image_data.shape[:2]
                print(f"Image size: {img_width}x{img_height}")
                split_count = max(1, img_width // 256)  # 动态计算分割数
                if split_count < 1:
                    print(f"Skipping - image too small (width < 256)")
                    continue

                # 处理图像分割
                image_name = os.path.splitext(filename)[0]
                rel_path = os.path.relpath(dirpath, root_dir)  # 相对路径作为文件名一部分
                for i in range(split_count):
                    start = i * 256
                    end = (i + 1) * 256
                    if end > img_width:
                        end = img_width
                        start = max(0, end - 256)
                    if end - start < 128:
                        print(f"Skipping small patch {i} (size: {end-start}px)")
                        continue

                    normal_images.append(image_data[:, start:end, :])
                    normal_fname.append(f"{rel_path}_{image_name}_{i}".replace(os.sep, '_'))  # 替换路径分隔符
                    image_count += 1

        print(f"\nRecursive processing complete. Total files checked: {file_count}, valid images processed: {image_count}")
        return image_count

    # 处理正常图像（递归遍历所有子目录）
    normal_root = os.path.join(args.dataset_root, 'NODefect_images')
    print(f"Starting normal image processing from root: {normal_root}")
    processed_normal_count = process_normal_images(normal_root)

    # 处理异常图像
    outlier_root = os.path.join(args.dataset_root, 'Defect_images/Defect_images')
    label_root = os.path.join(args.dataset_root, 'Mask_images/Mask_images')
    processed_defect_count = 0

    print(f"\nProcessing defect images from {outlier_root}...")
    if os.path.exists(outlier_root):
        for dirpath, _, filenames in os.walk(outlier_root):
            for image in tqdm(filenames, desc="Processing defect files"):
                image_path = os.path.join(dirpath, image)
                if not os.path.isfile(image_path) or not is_image_file(image_path):
                    continue

                image_name = os.path.splitext(image)[0]
                label_path = os.path.join(label_root, f"{image_name}_mask.png")
                if not os.path.exists(label_path):
                    continue

                image_data = cv2.imread(image_path)
                label_data = cv2.imread(label_path)
                if image_data is None or label_data is None:
                    continue

                if image_data.shape[:2] != label_data.shape[:2]:
                    continue

                img_height, img_width = image_data.shape[:2]
                split_count = max(1, img_width // 256)
                for i in range(split_count):
                    start = i * 256
                    end = (i + 1) * 256
                    if end > img_width:
                        end = img_width
                        start = max(0, end - 256)

                    im_patch = image_data[:, start:end, :]
                    la_patch = label_data[:, start:end, :]
                    if np.max(la_patch) != 0:
                        outlier_images.append(im_patch)
                        outlier_labels.append(la_patch)
                        outlier_fname.append(f"{image_name}_{i}")
                        processed_defect_count += 1
    else:
        print(f"Defect root directory not found: {outlier_root}")

    # 保存处理结果
    if normal_images:
        normal_train, normal_test, normal_name_train, normal_name_test = train_test_split(
            normal_images, normal_fname, test_size=0.25, random_state=42
        )

        target_root = './AITEX_anomaly_detection/AITEX'
        os.makedirs(os.path.join(target_root, 'train/good'), exist_ok=True)
        os.makedirs(os.path.join(target_root, 'test/good'), exist_ok=True)
        os.makedirs(os.path.join(target_root, 'ground_truth'), exist_ok=True)

        # 保存训练集
        for img, name in tqdm(zip(normal_train, normal_name_train), desc="Saving train good"):
            cv2.imwrite(os.path.join(target_root, 'train/good', f"{name}.png"), img)

        # 保存测试集正常图像
        for img, name in tqdm(zip(normal_test, normal_name_test), desc="Saving test good"):
            cv2.imwrite(os.path.join(target_root, 'test/good', f"{name}.png"), img)

        # 保存异常图像和掩码
        for img, label, name in tqdm(zip(outlier_images, outlier_labels, outlier_fname), desc="Saving defects"):
            try:
                class_code = name.split('_')[1]
                defect_class = DEFEAT_CLASS.get(class_code, "unknown")
                defect_dir = os.path.join(target_root, 'test', defect_class)
                mask_dir = os.path.join(target_root, 'ground_truth', defect_class)
                os.makedirs(defect_dir, exist_ok=True)
                os.makedirs(mask_dir, exist_ok=True)
                cv2.imwrite(os.path.join(defect_dir, f"{name}.png"), img)
                cv2.imwrite(os.path.join(mask_dir, f"{name}_mask.png"), label)
            except Exception as e:
                print(f"Error saving {name}: {e}")

        print(f"\nProcessing Summary:")
        print(f"- Normal train: {len(normal_train)}")
        print(f"- Normal test: {len(normal_test)}")
        print(f"- Defect test: {len(outlier_images)}")
    else:
        print("\nNo normal images processed! Detailed check:")
        print(f"1. Normal root exists? {os.path.exists(normal_root)}")
        print(f"2. Normal root is a directory? {os.path.isdir(normal_root)}")
        print(f"3. Number of files checked in normal root: {processed_normal_count}")
        print(f"   If this is 0, check if the directory is empty or has no accessible files")
        print(f"4. Check if files have read permissions (try opening manually)")

if __name__ == "__main__":
    main()