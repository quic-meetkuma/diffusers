#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
from contextlib import nullcontext
import matplotlib.pyplot as plt

import torch
import torch_qaic
import torch.utils.checkpoint
from transformers import PretrainedConfig

from diffusers import (
    AutoencoderKL,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.training_utils import (
    free_memory,
)
from diffusers.utils import (
    check_min_version,
)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.36.0.dev0")

# logger = get_logger(__name__)
import logging
logger = logging.getLogger('diffuser')


def load_text_encoders(class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant, 
        # device_map=text_encoder_three_device_map
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def log_validation(
    pipeline,
    args,
    pipeline_args,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device="cpu").manual_seed(args.seed) if args.seed is not None else None
    autocast_ctx = nullcontext()

    with autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator, num_inference_steps=args.denoising_steps, height=args.resolution, width=args.resolution).images[0] for _ in range(args.num_validation_images)]

    del pipeline
    free_memory()

    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        default=24,
        help="Number of denoising steps for image generation.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    weight_dtype = torch.float16

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant, 
        # device_map=transformer_device_map
    )
    text_encoder_one.to(weight_dtype)
    text_encoder_two.to(weight_dtype)
    text_encoder_three.to(weight_dtype)
    vae.to(weight_dtype)

    te_1_device_id = 0
    te_2_device_id = 0
    te_3_device_id = 0
    vae_device_id = 0
    transformer_device_id = 1
    text_encoder_one.to(f"qaic:{te_1_device_id}", dtype=weight_dtype)
    text_encoder_two.to(f"qaic:{te_2_device_id}", dtype=weight_dtype)
    text_encoder_three.to(f"qaic:{te_3_device_id}", dtype=weight_dtype)
    vae.to(f"qaic:{vae_device_id}", dtype=weight_dtype)
    transformer.to(f"qaic:{transformer_device_id}", dtype=weight_dtype)
    
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        text_encoder_3=text_encoder_three,
        transformer=transformer,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    # load attention processors
    if args.output_dir is None:
        print("Using pretrained model as no LoRA adapters are provided using --output_dir flag.")
        generated_img_name = "original_results_"
    else:
        print("Using finetuned model using adapters provided through --output_dir flag.")
        pipeline.load_lora_weights(args.output_dir)
        generated_img_name = "finetuned_results_"
    pipeline_args = {"prompt": args.validation_prompt}

    # run inference
    if args.validation_prompt and args.num_validation_images > 0:
        pipeline_args = {"prompt": args.validation_prompt}
        images = log_validation(
            pipeline=pipeline,
            args=args,
            pipeline_args=pipeline_args,
        )
        for i, image in enumerate(images):    
            output_file = f"./{generated_img_name}{i}.png"
            plt.imsave(output_file, image)


if __name__ == "__main__":
    args = parse_args()
    main(args)
