import json
import re
from PIL import Image
import torch
import requests
from transformers import AutoModelForVision2Seq, AutoProcessor

from qwen_vl_utils import process_vision_info

from PIL import Image
import requests
import copy
import torch

import sys
import warnings
from instruction_generation import *
from peft import PeftModel
from loguru import logger
import tqdm
from vllm import LLM, SamplingParams
import os

# Set the environment variable
os.environ["VLLM_ATTENTION_BACKEND"] = "triton"

device = 'cuda'

model_path = "zhehuderek/praxis_vlm_7b_decisionmaking"


llm = LLM(
    model=model_path,
    limit_mm_per_prompt={"image": 1, "video": 0},
    max_model_len=120000,
    gpu_memory_utilization=0.5
)

sampling_params = SamplingParams(
    temperature=0.2,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=4096,
    n=1
)


# default processer
processor = AutoProcessor.from_pretrained(model_path)


def qwen2_inference(instruction, image_path, gen_num=1):
    SYSTEM_PROMPT="""You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST BE enclosed within <think> and </think> tags, and the final answer MUST BE enclosed within <answer> and </answer> tags."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": text,
        "multi_modal_data": mm_data,
    }
    print(f"llm_inputs: {llm_inputs}")
    outputs = llm.generate([llm_inputs]*gen_num, sampling_params=sampling_params)
    # generated_text = outputs[0].outputs[0].text
    # return generated_text
    generated_texts = [elem.text for elem in outputs[0].outputs]
    # print(f"generated_texts: {outputs}")
    return generated_texts



def extract_reason_and_answer(text):
    reason_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    answer = answer.replace("<answer>", "").replace("</answer>", "").strip().strip("*")
    return {"reason": reason, "answer": answer}


def formulate_instruction_mcq(sample_dict):
    option_str = "\n".join(sample_dict["action_list"])
    cur_input = f'''
You are given a situation and a question. \nBased on the situation provided, select the most appropriate option to answer the question:\n\n## Situation: \nShown in the given image.\n\n## Question:\nSelect the most appropriate course of initial action to take\n{option_str}\n\nNow answer the question. Just output the choice:'
'''
    return cur_input.strip()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, required=False)
    parser.add_argument('--write_path', type=str, required=False)

    args = parser.parse_args()
    read_path = args.read_path
    write_path = args.write_path

    # Download the data from the huggingface: zhehuderek/VIVA_Benchmark_EMNLP24
    # Download the images from the link: https://drive.google.com/drive/folders/1eFLdVoRdw3kNXdi-zkmf_NDyxEbbpGY5?usp=drive_link
    read_path = "VIVA_annotation.json"
    image_folder = "viva_images/"
    write_path = f"./viva_praxis_vlm_7b_vllm.json"

    print("write_path: ", write_path)

    data = json.load(open(read_path))
    data_pred = []
    for sample in tqdm.tqdm(data):
        cur_preds = []
        instruction = formulate_instruction_mcq(sample)
        image_path = image_folder + sample["image_file"]
        print(f"- prompt:\n{[instruction]}")
        outputs = qwen2_inference(instruction, image_path, gen_num=1)
        for output in outputs:
            print(f"- original output:\n{[output]}\n")
            if "</think>" in output and "<think>" in output:
                output = extract_reason_and_answer(output)
                print(f"- parsed output:\n{output}\n")
                sample["model_output"] = output["answer"]
                sample["reason"] = output["reason"]
            else:
                print(f"- output:\n{output}\n")
                sample["model_output"] = output
            cur_preds.append({"instruction": instruction, "prediction": output})
        sample["result"] = cur_preds
        data_pred.append(sample)
    
    with open(write_path, "w") as f_w:
        json.dump(data_pred, f_w, indent=2)

if __name__ == "__main__":

    main()
    
