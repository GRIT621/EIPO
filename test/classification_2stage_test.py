import json
file_path = r"./class_dir/contect.json"

data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()  # 去掉多余换行符
        if line:  # 避免空行
            data.append(json.loads(line))
# 提取字段
texts = [item["context"] for item in data]
labels = [item["choice"] for item in data]
subjects = [item["subject"] for item in data]

from Agent.classification_2stage_agent import TextAgent

agent = TextAgent(
    name="EmotionAgent",
    model="pro-deepseek-r1",  # 或你自己的 "pro-deepseek-v3"
    api_key="",
    api_base=""  # 或者你自己的base
)

results = agent.process_texts(
    texts=texts,
    labels=labels,  # ✅ 这里传 labels 而不是 custom_prompt
    subjects=subjects,
    save_path=r"./result/emotion_class_results_deepseekr1_3point_prompt.jsonl"
)