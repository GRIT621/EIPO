import json
file_path = r"./dataset/Conversations_Long.jsonl"

data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()  # 去掉多余换行符
        if line:  # 避免空行
            data.append(json.loads(line))
# 提取字段
conversation_history = [item["conversation_history"] for item in data]

# print(texts[0])
from Agent.conversation_agent_newprompt import ConversationAgent

# agent = ConversationAgent(
#     name="ConversationAgent",
#     model="pro-deepseek-r1",  # 或你自己的 "pro-deepseek-v3"
#     api_key="sk-W1dB5IOIuIunzR0bJLkxNiKU46SQD2JS4vhqbDL3E55cgr3V",
#     api_base="https://api.probex.top/v1/"  # 或者你自己的base
# )
# 初始化Agent
agent = ConversationAgent(
    name="ConversationAgent",
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
    texts=conversation_history,
    save_path=r"./result/conversation_results_r1_newprompt.jsonl"
)

# 4️⃣ 输出示例结果
print("✅ 全部处理完成！示例输出：")
print(results[0])

