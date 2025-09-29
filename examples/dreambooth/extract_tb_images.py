import matplotlib.pyplot as plt
from tbparse import SummaryReader  
import glob
import os

tb_log_dir = "./trained-sd3-lora_512x512_v3/tb_logs"
tb_log_files = glob.glob(os.path.join(tb_log_dir) + "*")

for tb_log_file in tb_log_files:
    reader = SummaryReader(tb_log_file) 
    df = reader.images
    if len(df) == 0:
        continue
    for index, row in df.iterrows():
        tag = row['tag']
        value = row['value']
        step = row['step']
        img_name = f"{tag}_{step}.png"
        output_file = os.path.join(tb_log_dir, img_name)
        plt.imsave(output_file, value)
