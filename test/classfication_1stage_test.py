import json
file_path = r"./dataset//Emotion_Classification.jsonl"

data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()  # 去掉多余换行符
        if line:  # 避免空行
            data.append(json.loads(line))
# 提取字段
texts = [item["context"] for item in data]
labels = [item["choices"] for item in data]
subjects = [item["Subject"] for item in data]

# print(texts[0])
from Agent.classification_2stage_agent import TextAgent


# 初始化Agent
agent = TextAgent(
    name="EmotionAgent",
    model="pro-deepseek-v3",  # 或你自己的 "pro-deepseek-v3"
    api_key="",
    api_base=""  # 或者你自己的base
)


# 3️⃣ 运行主流程
results = agent.process_texts(
    texts=texts,
    labels=labels,  # ✅ 这里传 labels 而不是 custom_prompt
    subjects=subjects,
    save_path=r"./result/classification_results_deepseekv3_en_sg2.log"
)

# 4️⃣ 输出示例结果
print("✅ 全部处理完成！示例输出：")
print(results[0])

