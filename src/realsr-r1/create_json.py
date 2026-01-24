# import os
# import json

# image_folder = "/work/docker/GRPO/dataset/dataset_image_LR_v2"
# text_folder = "/work/docker/GRPO/dataset/dataset_tag"

# # 获取文件夹内所有文件并排序
# image_files = sorted(os.listdir(image_folder)) 
# text_files = sorted(os.listdir(text_folder))

# # 初始化列表存储JSON数据
# data = []

# # 遍历所有图像文件
# for image_file, text_file in zip(image_files, text_files):
#     # 确保文件是有效的图片和文本文件
#     if image_file.endswith('.jpg') or image_file.endswith('.png'):
#         # 构建图片路径
#         image_path = os.path.join(image_folder, image_file)
        
#         # 构建文本文件路径
#         text_path = os.path.join(text_folder, text_file)
        
#         # 读取文本文件内容，假设文本文件包含问题和答案
#         with open(text_path, 'r') as f:
#             tag = f.readlines()

  
        
#         # 将信息添加到data列表
#         data.append({
#             'image_path': image_path,
#             'problem': 'Perceive the degradation, understand the image content, and restore the high-quality image step by step (simulating the image restoration process from coarse to fine). The low-quality image is as follows: <|image|>. The generation format should be as follows: <degradation> ... </degradation> <rough_understand> ... </rough_understand> <rough_image> ... </rough_image> <middle_understand> ... </middle_understand> <middle_image> ... </middle_image> <final_understand> ... </final_understand> <final_image> ... </final_image>.',
#             'tag': tag
#         })

# # 将数据写入到JSON文件
# json_file_path = "/work/docker/GRPO/grpo-lumina/src/virft/src/open_r1/data_v2.json"
# with open(json_file_path, 'w') as json_file:
#     json.dump(data, json_file, indent=4)

# print(f"JSON file saved to {json_file_path}")

import time
from datasets import DatasetDict, Dataset
from PIL import Image
import json

"""
turn your json to DatasetDict
"""
def json_to_dataset(json_file_path):
    # read json file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    image_paths = [item['image_path'] for item in data]
    problems = [item['problem'] for item in data]
    tags = [item['tag'] for item in data]

    images = [Image.open(image_path).convert('RGBA') for image_path in image_paths]

    dataset_dict = {
        'image': images,
        'problem': problems,
        'tag': tags
    }

    dataset = Dataset.from_dict(dataset_dict)
    dataset_dict = DatasetDict({
        'train': dataset
    })
    return dataset_dict


time1 = time.asctime()
print(time1)
### Your dataset in JSON file format consists of three parts: image, problem and solution
dataset_dict = json_to_dataset('/work/docker/GRPO/grpo-lumina/src/virft/src/open_r1/data_v2.json')
time2 = time.asctime()
print(time2)

"""
save to your local disk
"""
def save_dataset(dataset_dict, save_path):
    # save DatasetDict to your disk
    dataset_dict.save_to_disk(save_path)

save_path = '/work/docker/GRPO/grpo-lumina/src/virft/src/open_r1/test500_v2'
save_dataset(dataset_dict, save_path)