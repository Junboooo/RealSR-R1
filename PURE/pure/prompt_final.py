import os
import json

# 定义目录路径
a_dir = '/root/wx1233510/data/training_for_ar_LSFF/description_final'  # degradation txt所在目录
b_dir = [
    '/root/wx1233510/data/training_for_ar_LSFF/llava_4_0417',
    '/root/wx1233510/data/training_for_ar_LSFF/llava_2_0417',
    '/root/wx1233510/data/training_for_ar_LSFF/llava_0417',
]  # understanding txt所在目录
image_dirs = [
    '/root/wx1233510/data/training_for_ar_LSFF/lr',  # 图像所在目录
    '/root/wx1233510/data/training_for_ar_LSFF/gt_4',
    '/root/wx1233510/data/training_for_ar_LSFF/gt_2',
    '/root/wx1233510/data/training_for_ar_LSFF/gt'
]

# 获取所有txt文件（000001.txt 到 001000.txt）
und_files = [f for f in os.listdir(b_dir[0]) if f.endswith('.txt')]  # 假设所有目录下的文件一致
und_files.sort()

# 构建字典
data_dict = []

for und_file in und_files:
    # 获取degradation文件的内容
    deg_file = und_file.replace('.txt', '.txt')  # 假设degradation文件与understanding文件名称相同
    with open(os.path.join(a_dir, deg_file), 'r') as deg_f:
        deg = deg_f.read().strip()

    # 获取understanding文件的内容，并合并所有理解文件
    und_content = []
    for und_dir in b_dir:
        with open(os.path.join(und_dir, und_file), 'r') as und_f:
            und = und_f.read().strip()
            und_content.append(und)
    
    # import pdb;pdb.set_trace()

    # 获取图像路径
    image_paths = [
        os.path.join(image_dirs[0], os.path.basename(deg_file).replace('.txt', '.png')),  # lr
        os.path.join(image_dirs[1], os.path.basename(deg_file).replace('.txt', '.png')),  # hr_4
        os.path.join(image_dirs[2], os.path.basename(deg_file).replace('.txt', '.png')),  # hr_2
        os.path.join(image_dirs[3], os.path.basename(deg_file).replace('.txt', '.png'))   # hr
    ]
    
    # 构建对话内容
    conversation = [
        {
            "from": "human",
            "value": "Perceive the degradation, understand the image content, and restore the high-quality image step by step. <|image|>"
        },
        {
            "from": "gpt",
            "value": f"{deg}. {und_content[0]} <|image|> {und_content[1]} <|image|> {und_content[2]} <|image|>"
        }
    ]
    
    # 构建字典并添加到列表
    data_dict.append({
        "conversations": conversation,
        "image": image_paths
    })

# 保存结果到JSON文件
output_file = '/root/wx1233510/data/training_for_ar_LSFF/prompt_final_pipline.json'
with open(output_file, 'w') as json_file:
    json.dump(data_dict, json_file, indent=2)

print(f"结果已保存至 {output_file}")
