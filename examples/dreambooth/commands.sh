export MODEL_NAME="stabilityai/stable-diffusion-3.5-large-turbo"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-sd3-lora"

QAIC_VISIBLE_DEVICES=32 accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=256 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=4e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --num_validation_images=1 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" 


QAIC_VISIBLE_DEVICES=32,33,34,35,36,37,38,39 accelerate launch --config_file ddp_config.yaml \
  train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=256 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=4e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --num_validation_images=1 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" 


export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-sd3-lora_256x256_5xddp"
QAIC_VISIBLE_DEVICES=32,33,34,35,36,37,38,39,40,41 python \
  train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=256 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=5 \
  --learning_rate=4e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100 \
  --num_validation_images=1 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0"


cd /home/meetkuma/diffusers_exp/diffusers/examples/dreambooth/
source /home/meetkuma/python_env/diffuser_env/bin/activate
export MODEL_NAME="stabilityai/stable-diffusion-3.5-large-turbo"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-sd3-lora_64x64_v3"
export HF_HOME=/home/meetkuma/tmp

ACCELERATE_BYPASS_DEVICE_MAP="true" DEVICES_PER_RANK=2 \
TE_1_DEVICE_ID=0 TE_2_DEVICE_ID=0 TE_3_DEVICE_ID=0 \
VAE_DEVICE_ID=0 TRANSFORMER_DEVICE_ID=1 \
QAIC_VISIBLE_DEVICES=0,1 DEBUG_LOGS=1 python \
  train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=4e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=25 \
  --num_validation_images=1 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=5 \
  --seed="0" --enable_profiling



cd /home/meetkuma/diffusers_exp/diffusers/examples/dreambooth/
source /home/meetkuma/python_env/diffuser_env/bin/activate
export MODEL_NAME="stabilityai/stable-diffusion-3.5-large-turbo"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-sd3-lora_512x512_v3"
export HF_HOME=/home/meetkuma/tmp

# Original model inference
QAIC_VISIBLE_DEVICES=0,1 python infer_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --resolution=512 \
  --num_validation_images=5 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --denoising_steps 24 --seed="0"

# Finetuned model inference
QAIC_VISIBLE_DEVICES=0,1 python infer_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --num_validation_images=5 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --denoising_steps 24 --seed="0"
