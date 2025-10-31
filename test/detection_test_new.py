import json
file_path = r"/Users/grit/PycharmProjects/LongEmo/dataset/Emotion_Detection.jsonl"

data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()  # 去掉多余换行符
        if line:  # 避免空行
            data.append(json.loads(line))
# 提取字段
texts = [item["text"] for item in data]
# print("texts[0]:",texts[0])



# print(texts[0])
from Agent.detection_agent_new import DetectionAgent


# 初始化Agent
agent = DetectionAgent(
    name="EmotionAgent",
    model="pro-deepseek-r1",  # 或你自己的 "pro-deepseek-v3"
    api_key="sk-dasvprtzlkkwmkra",
    api_base="https://cloud.infini-ai.com/maas/v1"  # 或者你自己的base
)
# 2️⃣ 初始化 Agent
# agent = TextAgent(
#     name="EmotionAgent",
#     model="pro-deepseek-v3",
#     api_key="sk-xxxxxx",  # 替换成你自己的 key
#     api_base="https://cloud.infini-ai.com/maas/v1"
# )

# 3️⃣ 运行主流程
results = agent.process_texts(
    texts=texts,
    save_path=r"./result/detection_results_r1_new.jsonl"
)

# 4️⃣ 输出示例结果
print("✅ 全部处理完成！示例输出：")
print(results[0])

