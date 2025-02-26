import os
import random
import glob

# 设置路径
source_dir = "data/nuscenes/customer/pivot-bezier"
output_dir = "assets/splits/nuscenes"
train_file = os.path.join(output_dir, "train.txt")
val_file = os.path.join(output_dir, "val.txt")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取所有文件名（不包括后缀）
files = glob.glob(os.path.join(source_dir, "*"))
filenames = [os.path.splitext(os.path.basename(f))[0] for f in files]

# 设置随机种子以确保结果可重复
random.seed(42)

# 随机打乱文件名列表
random.shuffle(filenames)

# 计算分割点
split_idx = int(len(filenames) * 0.8)

# 分割为训练集和验证集
train_filenames = filenames[:split_idx]
val_filenames = filenames[split_idx:]

# 输出到文件
with open(train_file, 'w') as f:
    f.write('\n'.join(train_filenames))

with open(val_file, 'w') as f:
    f.write('\n'.join(val_filenames))

print(f"总文件数: {len(filenames)}")
print(f"训练集文件数: {len(train_filenames)}")
print(f"验证集文件数: {len(val_filenames)}")
print(f"训练集文件已保存至: {train_file}")
print(f"验证集文件已保存至: {val_file}")