import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# ===================== 【你的路径】 =====================
SOURCE_ROOT = r"D:\课程设计\EgoExo-Fitness\only_exo_m"
TARGET_ROOT = r"D:\课程设计\EgoExo-Fitness\small_dataset"
MAX_PER_FOLDER = 2000  # 每个文件夹最多 2000 张
# ======================================================

# 新建输出目录
all_images = []

# 遍历每个视频文件夹
for video_folder in os.listdir(SOURCE_ROOT):
    video_path = os.path.join(SOURCE_ROOT, video_folder)
    exo_m_path = os.path.join(video_path, "exo_m")

    if not os.path.exists(exo_m_path):
        continue

    # 获取该文件夹所有图片
    imgs = sorted([f for f in os.listdir(exo_m_path) if f.endswith(".jpg")])
    selected = imgs[:MAX_PER_FOLDER]

    print(f"📂 {video_folder} 取前 {len(selected)} 张")

    # 记录完整路径
    for img in selected:
        img_path = os.path.join(exo_m_path, img)
        all_images.append(img_path)

# 8:2 划分
train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
print(f"\n✅ 总计：训练集 {len(train_imgs)} 张 | 验证集 {len(val_imgs)} 张")

# 复制文件（保持 图片+标签 一起复制）
def copy_files(img_list, split):
    for src_img in img_list:
        # 目标路径
        name = os.path.basename(src_img)
        name_no_ext = os.path.splitext(name)[0]
        dst_img = os.path.join(TARGET_ROOT, split, "images", name)
        dst_txt = os.path.join(TARGET_ROOT, split, "labels", name_no_ext + ".txt")
        src_txt = os.path.splitext(src_img)[0] + ".txt"

        # 创建目录
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        os.makedirs(os.path.dirname(dst_txt), exist_ok=True)

        # 复制
        shutil.copy2(src_img, dst_img)
        if os.path.exists(src_txt):
            shutil.copy2(src_txt, dst_txt)

# 开始复制
copy_files(train_imgs, "train")
copy_files(val_imgs, "val")

print("\n🎉 小数据集构建完成！")
print(f"📂 路径：{TARGET_ROOT}")