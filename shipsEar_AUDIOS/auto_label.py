import os
import shutil

# 国际顶会通用的 ShipsEar 5大类标准映射字典 (根据前缀数字)
# Class A: 疏浚船、拖网渔船、贻贝船、拖船 (Mussel boats, Dredgers, Trawlers, Tugboats)
# Class B: 摩托艇、领航船、帆船 (Motorboats, sailboats, pilot boats)
# Class C: 客轮、渡轮 (Passenger ferries)
# Class D: 远洋班轮、滚装船 (Ocean liners, ro-ro vessels)
# Class E: 背景噪声 (Background noise)

class_mapping = {
    'A': [15, 28, 31, 46, 47, 48, 49, 66, 73, 75, 76, 80, 93, 94, 95, 96],
    'B': [21, 26, 27, 29, 30, 33, 37, 39, 45, 50, 51, 52, 56, 57, 68, 70, 72, 77, 79],
    'C': [6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 32, 34, 35, 36, 38, 40, 41, 42, 43, 53, 54, 55, 59, 60, 61, 62, 63, 64, 65, 67],
    'D': [16, 18, 19, 20, 22, 23, 24, 25, 58, 69, 71, 78],
    'E': [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92],
}
# 剩下的数字全部默认归入 E 类（背景噪声）

# 创建分类文件夹
for cls in ['A', 'B', 'C', 'D', 'E']:
    os.makedirs(cls, exist_ok=True)

# 遍历当前目录下的所有 wav 文件
wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]

for file_name in wav_files:
    # ShipsEar 的文件命名规则通常是: 编号__日期_时间.wav (例如 6__04_02_13.wav)
    try:
        # 提取文件名前面的数字编号
        file_num = int(file_name.split('_')[0])
        
        target_class = 'E' # 默认背景噪声
        for cls, num_list in class_mapping.items():
            if file_num in num_list:
                target_class = cls
                break
                
        # 移动文件到对应的分类文件夹
        shutil.move(file_name, os.path.join(target_class, file_name))
        print(f"成功将 {file_name} 移动至标签 {target_class} 类文件夹")
        
    except ValueError:
        print(f"跳过不符合命名规则的文件: {file_name}")

print("所有音频自动分类打标签完成！")

# # 1. 把所有子文件夹里的 wav 文件移动到当前目录
# mv A/*.wav B/*.wav C/*.wav D/*.wav E/*.wav . 2>/dev/null

# # 2. 彻底删除旧的文件夹
# rm -rf A B C D E