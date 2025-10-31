import json
import re

def extract_predicted_index_from_group(group_lines):
    """
    按最终规则提取 predicted_index：
    1. $$$$$most_diff: 如果唯一
    2. @@@@@rare_candidates: 如果most_diff不唯一且rare唯一
    3. !!!!max_abs_candidate: 如果rare不唯一且max唯一
    4. Final decision: 否则取这里的值
    """
    most_diff = None
    rare_candidates = None
    max_abs_candidate = None
    final_decision = None

    for line in group_lines:
        # 提取 most_diff
        if line.strip().startswith("$$$$$most_diff:"):
            match = re.search(r'\[(.*?)\]', line)
            if match:
                most_diff = [int(x.strip()) for x in match.group(1).split(',') if x.strip().isdigit()]

        # 提取 rare_candidates
        if line.strip().startswith("@@@@@rare_candidates:"):
            match = re.search(r'\[(.*?)\]', line)
            if match:
                rare_candidates = [int(x.strip()) for x in match.group(1).split(',') if x.strip().isdigit()]

        # 提取 max_abs_candidate
        if line.strip().startswith("!!!!max_abs_candidate:"):
            match = re.search(r'\[(.*?)\]', line)
            if match:
                max_abs_candidate = [int(x.strip()) for x in match.group(1).split(',') if x.strip().isdigit()]

        # 提取 Final decision
        if "Final decision" in line:
            match = re.search(r'Final decision[:：]\s*(\d+)', line)
            if match:
                final_decision = int(match.group(1))

    # 规则判断顺序：
    if most_diff and len(most_diff) == 1:
        return most_diff[0]
    if most_diff and len(most_diff) > 1:
        if rare_candidates and len(rare_candidates) == 1:
            return rare_candidates[0]
        if rare_candidates and len(rare_candidates) > 1:
            if max_abs_candidate and len(max_abs_candidate) == 1:
                return max_abs_candidate[0]
            if final_decision is not None:
                return final_decision

    return None  # 若无法决策，返回None


def parse_log_file(input_file, output_file):
    results = []
    group_lines = []
    current_id = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 检测新的组开头（例如：🔹 正在处理第 X/XXX 组文本）
            if line.strip().startswith("🔹 正在处理第"):
                # 如果已经有上一组内容，进行解析
                if group_lines:
                    predicted = extract_predicted_index_from_group(group_lines)
                    print("!!!!!",group_lines[0][:14], predicted)
                    results.append({"id": current_id, "predicted_index": predicted})
                    current_id += 1
                    group_lines = []
            group_lines.append(line)

        # 文件结束后最后一组也要解析
        if group_lines:
            predicted = extract_predicted_index_from_group(group_lines)
            results.append({"id": current_id, "predicted_index": predicted})

    # 输出为 jsonl 文件
    with open(output_file, 'w', encoding='utf-8') as fw:
        for item in results:
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 解析完成，共写入 {len(results)} 条至 {output_file}")


if __name__ == "__main__":
    input_file = "/result/detection_result_2_stage_campare2.log"  # 输入日志文件
    output_file = "/Users/grit/PycharmProjects/LongEmo/final json/detection_json_deepseekr1.jsonl"  # 输出结果文件
    parse_log_file(input_file, output_file)
