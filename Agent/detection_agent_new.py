# 文件名：text_agent.py
from base_agent import BaseAgent
from typing import List
import json
import os
from textwrap import dedent
from openai import OpenAI

class DetectionAgent(BaseAgent):
    """针对文本数据调用大模型的Agent（判断乐观/悲观情绪并检测极端者）"""

    def __init__(self, name: str, model: str, api_key: str, api_base: str):
        super().__init__(name, model, api_key, api_base)

    def process_batch(self, batches, index_to_text,client):
        print("@@@@",batches)
        all_scores = [s for b in batches for s in b["scores"]]
        # most_diff = [d for b in batches for d in b["most_different_ids"]]
        most_diff = [batch["most_different_id"] for batch in batches]
        # most_diff = [d for b in batches for d in b["most_different_id"]]

        print("$$$$$most_diff:",most_diff)
        pos_count = sum(1 for s in all_scores if s["label"] == "optimistic")
        neg_count = sum(1 for s in all_scores if s["label"] == "pessimistic")
        print(f"Positive count: {pos_count}, Negative count: {neg_count}")
        # 判断稀缺情绪
        if pos_count > neg_count:
            scarce_label = 'pessimistic'
        elif pos_count < neg_count:
            scarce_label = 'optimistic'
        else:
            print('数量相等，无稀缺情绪，需要继续API')
            return

        print(f"Scarce emotion determined: {scarce_label}")

        id_to_label = {s["index"]: s["label"] for s in all_scores}
        id_to_score = {s["index"]: s["score"] for s in all_scores}

        # 5. 找出哪些 most_different_id 是稀缺情绪
        rare_candidates = [mid for mid in most_diff if id_to_label.get(mid) == scarce_label]
        print("@@@@@rare_candidates:",rare_candidates)
        # 6. 判断返回逻辑
        if len(rare_candidates) == 1:
            candidate_id = rare_candidates[0]
            candidate_score = id_to_score.get(candidate_id)
            candidate_label = id_to_label.get(candidate_id)
            print(f"Rare candidate found - ID: {candidate_id}, Score: {candidate_score}, Label: {candidate_label}")
            return candidate_id
        else:
        # 打印所有候选者的 id 和 score
            print("Multiple rare candidates found:")
            for cid in rare_candidates:
                print(f"ID: {cid}, Score: {id_to_score.get(cid)}, Label: {id_to_label.get(cid)}")

            # 找出 score 绝对值最大的候选者
            max_abs = max(abs(id_to_score[c]) for c in rare_candidates)
            max_abs_candidate = [c for c in rare_candidates if abs(id_to_score[c]) == max_abs]
            print("!!!!max_abs_candidate:",max_abs_candidate)
            if len(max_abs_candidate) == 1:
                candidate_id = max_abs_candidate[0]

                max_score = id_to_score.get(candidate_id)
                max_label = id_to_label.get(candidate_id)

                print(
                    f"Candidate with max absolute score - ID: {max_abs_candidate}, Score: {max_score}, Label: {max_label}")
                return max_abs_candidate
            else:
                texts_for_api = [index_to_text[c] for c in max_abs_candidate]
                print("Multiple candidates with same max absolute score:")
                for c in max_abs_candidate:
                    text = index_to_text[c]
                    score = id_to_score.get(c)
                    label = id_to_label.get(c)
                    print(f"  ID: {c}, Score: {score}, Label: {label}, Text: {text}")

                prompt = f"""
                        Please read the following texts and determine which one expresses the **most extreme emotion** 
                        toward COVID-19 (most {scarce_label}).
                        Return the index corresponding to the input list (0-based) of the texts below:
                        Texts:
                        {chr(10).join([f"{c}: {index_to_text[c]}" for c in max_abs_candidate])}
            
                        Return **only the original ID** of the most extreme one. 
                        Do not return any explanation, text, or additional characters.
                        """

                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a professional emotional psychology evaluator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    timeout=500
                )

                reply = response.choices[0].message.content.strip()
                print(f"Final decision: {reply}")
                return reply


    def process_texts(self, texts: List[List[dict]], save_path: str = None, batch_size: int = 19):
        """
        对每个文本组执行以下流程：
        - 分批发送文本（每批默认19条）
        - 对每个context给出乐观/悲观评分（-5~5）
        - 找出整体中情绪状态不同且最极端的那条，返回其 index
        """
        results = []
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        for i, text_blocks in enumerate(texts, start=1):
            print(f"\n🔹 正在处理第 {i}/{len(texts)} 组文本...")
            if i < 135:
                continue
            try:
                contexts = [block["context"] for block in text_blocks]
                indexes = [block["index"] for block in text_blocks]
                n_contexts = len(contexts)
                index_to_text = dict(zip(indexes, contexts))

                batch_start = 0
                batch_id = 0
                batch_results = []

                # ------------------------------
                # 分批发送
                # ------------------------------
                while batch_start < n_contexts:
                    batch_end = min(batch_start + batch_size, n_contexts)
                    batch_contexts = contexts[batch_start:batch_end]
                    batch_indexes = indexes[batch_start:batch_end]
                    batch_id += 1

                    print(f"  ▶️ 批次 {batch_id}: 发送 {batch_start}-{batch_end-1} 共 {len(batch_contexts)} 条")

                    # prompt 构造
                    prompt = dedent(f"""
                        Your task is to evaluate each person’s overall emotional state toward the COVID-19 pandemic.
                        Each person may have mixed emotions, but their **dominant overall attitude** is either **optimistic** or **pessimistic**.
                        For each text:
                        - Provide a **score from -5 to 5**, where:
                            -5 = extremely pessimistic,  
                            0 = neutral,  
                            +5 = extremely optimistic.
                        - Provide the corresponding label: “optimistic” or “pessimistic”.
                        
                        IMPORTANT INSTRUCTIONS:
                        - The dataset is designed such that the majority of people share the same overall emotional direction (either mostly optimistic or mostly pessimistic).
                        - At most **one** text may express the opposite dominant attitude.
                        - Mixed expressions (e.g., worry + confidence) must be judged by the *overall emotional direction*:
                            - If a person expresses concerns but still shows acceptance, resilience, or hope, classify as "optimistic" with a positive score.
                            - If a person expresses frustration, fear, or hopelessness even if they mention some facts neutrally, classify as "pessimistic" with a negative score.
                        - Avoid neutral scores unless the emotion is truly balanced with no clear direction.
                        - Do not cluster scores around zero. Use stronger positive or negative scores (+3, +4, -3, -4) to reflect the dominant attitude.

                        
                        After scoring all texts, determine:
                        - Which one is **most different** from the overall emotional trend (e.g., if most are optimistic, find the most pessimistic one, or vice versa).
                        - If all are the same, return “None”.
                        
                        Output format (strictly JSON):
                        {{
                            "scores": [
                                {{"index": int, "score": float, "label": "optimistic/pessimistic"}},
                                ...
                            ],
                            "most_different_id": int or "None"
                        }}

                        Here are the texts:
                    """).strip()

                    for idx, (c, real_id) in enumerate(zip(batch_contexts, batch_indexes), start=1):
                        prompt += f"\n\n[{real_id}] {c}"

                    # 调用模型
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a professional emotional psychology evaluator."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        timeout=500
                    )


                    reply = response.choices[0].message.content.strip()
                    print(f"✅ 批次 {batch_id} 完成")

                    # 尝试解析 JSON
                    try:
                        import re
                        # 去掉 Markdown 包裹
                        if "```" in reply:
                            reply = re.sub(r"^```(json)?", "", reply.strip(), flags=re.IGNORECASE)
                            reply = re.sub(r"```$", "", reply.strip())
                            reply = reply.strip()

                        # 提取 JSON 内容
                        match = re.search(r"\{[\s\S]*\}", reply)
                        if match:
                            reply = match.group(0)

                        parsed = json.loads(reply)
                        batch_results.append(parsed)
                    except Exception as e:
                        print(f"[Warning] 批次 {batch_id} JSON 解析失败: {e}")
                        print(f"⚠️ 模型原始返回内容预览（前200字）:\n{reply[:200]!r}")
                        batch_results.append({"raw": reply, "error": str(e)})

                    batch_start = batch_end

                f_result = self.process_batch(batch_results, index_to_text=index_to_text,client = client)




                # ------------------------------
                # 合并结果
                # ------------------------------
                results.append({
                    "id": i,
                    "predicted_index": f_result
                })
                print(f"第{i}组结果：",batch_results)

            except Exception as e:
                print(f"[Error] 第 {i} 组文本处理失败: {e}")
                results.append({
                    "group_index": i,
                    "error": str(e)
                })

        # ------------------------------
        # 保存结果
        # ------------------------------
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        return results

