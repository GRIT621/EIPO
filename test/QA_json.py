import json
import re

# 输入文件路径
input_file = "/Users/grit/PycharmProjects/LongEmo/result/qa_emotion_results_deepseek_r1_en.log"
# 输出文件路径
output_file = "./qa_jsonl_deepseek_r1_en.jsonl"


with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# 使用正则切分任意连续的 - 行
qa_blocks = re.split(r'\n-{3,}\n', content)

# 先建立一个字典，用实际问题编号映射回答
id_to_answer = {}

for block in qa_blocks:
    block = block.strip()
    if not block:
        continue

    # 提取问题编号
    m = re.search(r"【第(\d+)个问题：", block)
    if m:
        question_id = int(m.group(1)) - 1  # id从0开始
    else:
        continue  # 没有找到编号的块跳过

    # 提取回答
    answer_line = ""
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("回答") or line.startswith("Answer"):
            answer_line = line.split("：", 1)[-1].strip()
            break

    id_to_answer[question_id] = answer_line if answer_line else "None"

# 最终输出 120 条，缺失的用 None
total_questions = 120
output_lines = []

for i in range(total_questions):
    answer = id_to_answer.get(i, "None")
    output_dict = {
        "id": i,
        "predicted_answer": answer
    }
    output_lines.append(json.dumps(output_dict, ensure_ascii=False))

# 写入 JSONL 文件
with open(output_file, "w", encoding="utf-8") as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"转换完成！输出文件: {output_file}")