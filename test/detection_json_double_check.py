import re

import re
from openai import OpenAI
# agent = DetectionAgent(
#     name="EmotionAgent",
#     model="pro-deepseek-r1",  # 或你自己的 "pro-deepseek-v3"
#     api_key="sk-dasvprtzlkkwmkra",
#     api_base="https://cloud.infini-ai.com/maas/v1"  # 或者你自己的base
# )
# 接口地址：https://api.mixrai.com
# sk-AkWKr0v706oQniwQC8Bf507f6c154556B8F836F8764e2360

def ask_chatgpt_for_most_negative(text,label):
    client = OpenAI(api_key="sk-AkWKr0v706oQniwQC8Bf507f6c154556B8F836F8764e2360", base_url="https://api.mixrai.com/v1")

    messages=[{"role": "user",     "content":
        f"""
    You are given two text options:
    {text}
    Your task is to determine which text is more {label}. 
    **Only output a single int: 0 if the first text is more {label}, 1 if the second text is more {label}.** 
    Do not write any explanations, comments, or extra characters. 
    Strictly return only 0 or 1.
    """}]

    # 调用模型
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        timeout=120
    )
    return response.choices[0].message.content.strip()

import re

def extract_predicted_index_from_multicandidate_block(group_lines):
    """
    只处理包含 'Multiple candidates with same max absolute score:' 的块
    提取每个候选的 Text 部分，调用 ChatGPT API 返回最消极的候选 ID
    """
    multi_candidate_block = []
    final_decision = None
    collect = False  # 是否开始收集 Multiple candidates

    for line in group_lines:
        line = line.rstrip()

        if "Multiple candidates with same max absolute score:" in line:
            collect = True
            continue  # 下一行才是候选信息

        # 收集候选行，直到遇到 Final decision 或者下一组
        if collect:
            if line.strip().startswith("Final decision"):
                collect = False
                match_id = re.search(r'Final decision[:：]\s*(\d+)', line)
                if match_id:
                    final_decision = int(match_id.group(1))
                continue
            if line.strip():  # 非空行
                multi_candidate_block.append(line)



    if multi_candidate_block:
        # 提取每行的 Text 部分
        text_list = []
        id_list = []
        for entry in multi_candidate_block:
            text_match = re.search(r'Text: (.*)', entry)
            id_match = re.search(r'ID:\s*(\d+)', entry)
            label_match = re.search(r'Label:\s*(\d+)', entry)
            if text_match and id_match:
                text_list.append(text_match.group(1).strip())
                id_list.append(int(id_match.group(1)))
        print("@@@",multi_candidate_block)
        if text_list:
            # 调用 ChatGPT API 判断哪个最消极，返回对应 ID
            selected_index = ask_chatgpt_for_most_negative(text_list,label_match)
            print("!!!",id_list[int(selected_index)])
            return id_list[int(selected_index)]  # 返回 ChatGPT 选出的 ID

    # 没有 Multiple candidates 时返回 final_decision
    return final_decision

import json
import re

def parse_log_file(input_file, output_file):
    results = []
    group_lines = []
    current_id = None  # 保存当前组号

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_strip = line.strip()
            # 检测新的组开头，例如：🔹 正在处理第 1/136 组文本
            match = re.match(r"🔹 正在处理第\s*(\d+)", line_strip)
            if match:
                # 如果上一组有内容，先处理上一组
                if group_lines and current_id is not None:
                    predicted = extract_predicted_index_from_multicandidate_block(group_lines)
                    results.append({
                        "id": current_id,
                        "predicted_index": predicted
                    })
                    print({
                        "id": current_id,
                        "predicted_index": predicted
                    })
                    group_lines = []  # 重置上一组
                current_id = int(match.group(1))  # 更新当前组号

            # 收集当前行到组
            group_lines.append(line)

        # 文件结束后，处理最后一组
        if group_lines and current_id is not None:
            predicted = extract_predicted_index_from_multicandidate_block(group_lines)
            results.append({
                "id": current_id,
                "predicted_index": predicted
            })
            print(results)

    # 写入 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as fw:
        for item in results:
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    input_file = "/Users/grit/PycharmProjects/LongEmo/final json/detection_result_2_stage"      # 输入日志文件
    output_file = "/Users/grit/PycharmProjects/LongEmo/final json/double_check_gpt3.5.jsonl"
    parse_log_file(input_file,output_file)
