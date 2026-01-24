import os
from glob import glob
from inference_solver import FlexARInferenceSolver
from PIL import Image

inference_solver = FlexARInferenceSolver(
    model_path="/root/wx1233510/PURE-main/PURE-main/pure/output/7B_ft/epoch1", #/root/wx1233510/Lumina-mGPT-main/lumina_mgpt/output/7B_lumina/epoch1-iter14999
    precision="bf16",
    target_size=512,
)

input_dir = "/root/wx1233510/Lumina-mGPT-main/lumina_mgpt/test_dataset/benchmark_drealsr/test_LR"
output_dir = "/root/wx1233510/PURE-main/PURE-main/result/benchmark_drealsr/PURE_ft_image"
os.makedirs(output_dir, exist_ok=True)  
# output_dir_2 = "/root/wx1233510/Lumina-mGPT-main/lumina_mgpt/test_dataset/benchmark_drealsr/7B_lumina_1024_3out_image_2"
# os.makedirs(output_dir_2, exist_ok=True)  
# output_dir_4 = "/root/wx1233510/Lumina-mGPT-main/lumina_mgpt/test_dataset/benchmark_drealsr/7B_lumina_1024_3out_image_4"
# os.makedirs(output_dir_4, exist_ok=True)  

text_output_dir = "/root/wx1233510/PURE-main/PURE-main/result/benchmark_drealsr/PURE_ft_text"
os.makedirs(text_output_dir, exist_ok=True)  

supported_extensions = {'*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff'}
for img_path in glob(os.path.join(input_dir, '*.*')):
    if not os.path.splitext(img_path)[1].lower() in [ext[1:] for ext in supported_extensions]:
        continue
    
    try:
        img = Image.open(img_path)
        qas = [["Perceive the degradation, understand the image content, and restore the high-quality image. <|image|>", None]]
        
        generated = inference_solver.generate(
            images=img,
            qas=qas,
            max_gen_len=8192,
            temperature=0.9,
            logits_processor=inference_solver.create_logits_processor(cfg=0.8, text_top_k=1),
        )

        a1 = generated[0]
        txt_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        txt_path = os.path.join(text_output_dir, txt_filename)
        with open(txt_path, 'w') as f:
            f.write(str(a1))

        # output_path_4 = os.path.join(output_dir_4, os.path.basename(img_path))
        # generated[1][0].save(output_path_4)
        # output_path_2 = os.path.join(output_dir_2, os.path.basename(img_path))
        # generated[1][1].save(output_path_2)
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        generated[1][0].save(output_path)
        print(f"Processed: {img_path} -> {output_path}")
    
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        continue
print("All images processed!")