import json
file_path = r"/Users/grit/PycharmProjects/LongEmo/dataset/Emotion_Summary.jsonl"

data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()  # 去掉多余换行符
        if line:  # 避免空行
            data.append(json.loads(line))
# 提取字段"consultation_process":
cause = [item["case_description"] for item in data]
consultation_process = [item["consultation_process"] for item in data]

# print(texts[0])
from summary_agent import SummaryAgent


# custom_prompt = """你是一个情感分析助手，请阅读以下文本，并判断其情绪类别：愤怒、悲伤、高兴、惊讶、害怕、厌恶、平静。
# 请仅输出一个类别名称。"""
#
# # 初始化Agent
agent = SummaryAgent(
    name="EmotionAgent",
    model="pro-deepseek-v3",  # 或你自己的 "pro-deepseek-v3"
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

# agent = TextAgent(
#     name="EmotionAgent",
#     model="pro-deepseek-v3",
#     api_key="sk-xxxxxx",  # 替换成你自己的 key
#     api_base="https://cloud.infini-ai.com/maas/v1"
# )

# 3️⃣ 运行主流程
results = agent.process_texts(
    cause=cause,
    consultation_process=consultation_process,  # ✅ 这里传 labels 而不是 custom_prompt
    save_path=r"/Users/grit/PycharmProjects/LongEmo/result/sum_emotion_results_deepseek_r1.log"
)

# 4️⃣ 输出示例结果
print("✅ 全部处理完成！示例输出：")
print(results[0])

