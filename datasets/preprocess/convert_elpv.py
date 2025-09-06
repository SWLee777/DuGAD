import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

# 原始数据集根目录（保持不变）
source_root = ""
# 图像文件夹路径
image_folder = os.path.join(source_root, "images")
# 标注文件路径
label_path = os.path.join(source_root, "labels", "labels.csv")

# 新的整理后数据集根目录（模仿MVTec结构）
target_root = ""
os.makedirs(target_root, exist_ok=True)

# 读取labels.csv（保持原读取方式）
data = np.genfromtxt(
    label_path,
    delimiter=",",
    dtype=None,
    names=True,
    encoding="utf-8"  # 明确指定编码
)

# 提取数据字段（模仿第二段代码的字段提取逻辑）
image_fnames = []
probs = []
types = []

for idx, row in enumerate(data):
    # 保持原图像文件名构造方式
    img_name = f"elpv_image_{idx}_{row['type']}.png"
    image_fnames.append(img_name)
    probs.append(row["probability"])
    types.append(row["type"])

# 转换为numpy数组便于处理（模仿第二段代码的数组操作）
image_fnames = np.array(image_fnames)
probs = np.array(probs)
types = np.array(types)

# 筛选正常样本和异常样本（模仿第二段代码的筛选逻辑）
normal_mask = probs == 0
normal_fnames = image_fnames[normal_mask]
normal_labels = probs[normal_mask]

# 异常样本筛选（联合条件）
outlier_mask = probs != 0
mask_mono = (types == 'mono') & outlier_mask
mask_poly = (types == 'poly') & outlier_mask

outlier_fnames_mono = image_fnames[mask_mono]
outlier_fnames_poly = image_fnames[mask_poly]

# 划分训练集和测试集（添加第二段代码的拆分逻辑）
normal_train, normal_test, _, _ = train_test_split(
    normal_fnames,
    normal_labels,
    test_size=0.25,
    random_state=42
)

# 创建训练集目录（仅放训练集正常样本）
train_good = os.path.join(target_root, "train", "good")
os.makedirs(train_good, exist_ok=True)
for img_name in normal_train:
    img_path = os.path.join(image_folder, img_name)
    if not os.path.exists(img_path):
        print(f"警告：图像文件 {img_name} 不存在，已跳过")
        continue
    shutil.copy(img_path, os.path.join(train_good, img_name))

# 创建测试集正常样本目录
test_good = os.path.join(target_root, "test", "good")
os.makedirs(test_good, exist_ok=True)
for img_name in normal_test:
    img_path = os.path.join(image_folder, img_name)
    if not os.path.exists(img_path):
        print(f"警告：图像文件 {img_name} 不存在，已跳过")
        continue
    shutil.copy(img_path, os.path.join(test_good, img_name))

# 创建mono异常样本目录
test_mono = os.path.join(target_root, "test", "mono")
os.makedirs(test_mono, exist_ok=True)
for img_name in outlier_fnames_mono:
    img_path = os.path.join(image_folder, img_name)
    if not os.path.exists(img_path):
        print(f"警告：图像文件 {img_name} 不存在，已跳过")
        continue
    shutil.copy(img_path, os.path.join(test_mono, img_name))

# 创建poly异常样本目录
test_poly = os.path.join(target_root, "test", "poly")
os.makedirs(test_poly, exist_ok=True)
for img_name in outlier_fnames_poly:
    img_path = os.path.join(image_folder, img_name)
    if not os.path.exists(img_path):
        print(f"警告：图像文件 {img_name} 不存在，已跳过")
        continue
    shutil.copy(img_path, os.path.join(test_poly, img_name))

# 保持ground_truth目录创建
os.makedirs(os.path.join(target_root, "ground_truth"), exist_ok=True)

# 输出统计信息
print("数据集转换完成，结构如下：")
print(f"训练集正常样本：{len(normal_train)} 个（存储在 train/good）")
print(f"测试集正常样本：{len(normal_test)} 个（存储在 test/good）")
print(f"测试集mono异常样本：{len(outlier_fnames_mono)} 个（存储在 test/mono）")
print(f"测试集poly异常样本：{len(outlier_fnames_poly)} 个（存储在 test/poly）")
