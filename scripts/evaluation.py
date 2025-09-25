import json
import os
import re
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from openai import OpenAI
import time
from loguru import logger


API_KEY = "Your_API_KEY" # Trustworthy
client = OpenAI(api_key=API_KEY)


def openai_generate(input_prompt, model="gpt-3.5-turbo", temperature=1):
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_prompt}
                ],
                temperature=temperature,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            break
        except Exception as e:
            print(["[OPENAI ERROR]: ", e])
            response = None
            time.sleep(5)
    if response != None:
    # print(response)
        response = response.choices[0].message.content
    return response


def build_temp_prompt(pred, acition_list):
    action_str = "\n".join(acition_list)
    prompt_temp = f'''You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer.
If the meaning of all options are significantly different from the answer, output Z. 
Options: {action_str}
Answer: {pred}
Now just output the option:
'''
    return prompt_temp.strip()



def extract_choice(text):
    # 1. Clean and normalize text
    text = text.upper()  # Convert to uppercase
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = text.replace('*', '')  
    # 2. Choice should not have uppercase letters before or after
    choices = re.findall(r'(?<![A-Z])([A-Z])(?=[\.\,\?\!\:\;]|$)', text)

    if not choices:
        return None

    # 3. If only one choice, return it directly
    if len(choices) == 1:
        return choices[0]

    # 4. If multiple choices, use heuristic rules
    choice_scores = {choice: 0 for choice in choices}

    # 4.1 Keywords around choices get points
    keywords = [
        '答案', '选择', '正确', '是', '对',
        'answer', 'correct', 'choose', 'select', 'right',
        '认为', '应该', '觉得', 'think', 'believe', 'should'
    ]

    # Get context for each choice (20 chars before and after)
    for choice in choices:
        pos = text.find(choice)
        context = text[max(0, pos-20):min(len(text), pos+20)]

        # Add points for keywords
        for keyword in keywords:
            if keyword.upper() in context:
                choice_scores[choice] += 1

        # Add points if choice is near the end (usually final answer)
        if pos > len(text) * 0.7:  # In last 30% of text
            choice_scores[choice] += 2

        # Add points if followed by punctuation
        if pos < len(text) - 1 and text[pos+1] in '。.!！,，':
            choice_scores[choice] += 1

    # Return highest scoring choice
    return max(choice_scores.items(), key=lambda x: x[1])[0]


def parse_mcq_answer(pred, acition_list):
    options = ["A", "B", "C", "D", "E"]
    parsed_pred = None
    if len(pred) < 5:
        for ch in pred:
            if ch in options:
                parsed_pred = ch
                break
    for option in options:
        if pred.startswith(option + "."):
            parsed_pred = option
            break

    if parsed_pred is None:
        parsed_pred = extract_choice(pred)

    if parsed_pred is None:
        temp_prompt = build_temp_prompt(pred[:300], acition_list)
        predict_answer = openai_generate(temp_prompt)
        if predict_answer is not None:
            for ch in predict_answer:
                if ch in options + ["Z"]:
                    parsed_pred = ch
                    break
    if parsed_pred:
        return parsed_pred
    else: 
        return "Z"


def extract_reason_and_answer(text):
    reason_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    answer = answer.replace("<answer>", "").replace("</answer>", "").strip().strip("*")
    return {"reason": reason, "answer": answer}


def eval_acc(data):
    # each elem is a tuple of (pred, answer)
    parsed_pred_list = []
    label_list_new = []
    for sample in data:
        option_list = sample["action_list"]
        label = sample["answer"]
        pred = sample["result"][0]["prediction"]
        if type(pred) is dict:
            pred = pred["answer"]
        if type(pred) is list:
            pred = pred[0]
        
        if "</think>" in pred and "<think>" in pred:
            pred = extract_reason_and_answer(pred)["answer"]

            
        label_list_new.append(label)

        if pred is None:
            parsed_pred_list.append("Z")
            continue
        if "USER" in pred and "\nASSISTANT:" in pred:
            pred = pred.split("\nASSISTANT:")[1].strip()
        parsed_pred = parse_mcq_answer(pred, option_list)
        parsed_pred_list.append(parsed_pred)
    print(len(parsed_pred_list), len(label_list_new))
    assert len(parsed_pred_list) == len(label_list_new)
    print(f"[LOG-pred]:  {parsed_pred_list[:10]}")
    print(f"[LOG-label]: {label_list_new[:10]}")
    
    acc = len([i for i in range(len(parsed_pred_list)) if parsed_pred_list[i] == label_list_new[i]]) / len(label_list_new)
    return acc
            

def eval_one(read_path):
    data = json.load(open(read_path))
    cur_acc = eval_acc(data)
    return cur_acc


if __name__ == "__main__":
    folder = "./results_viva/"
    all_files = [fil for fil in os.listdir(folder) if "json" in fil]
    all_files = sorted(all_files)
    for fil_name in all_files:
        print("File: ", fil_name)
        result = eval_one(folder + "/" + fil_name)
        print(f"result: {result}\n")





