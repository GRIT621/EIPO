# 文件名：text_agent.py
from base_agent import BaseAgent
from typing import List
import json
from textwrap import dedent


class SummaryAgent(BaseAgent):
    """针对文本数据逐条调用大模型的Agent（两阶段：无关段检测 + 情感标签预测）"""

    def __init__(self, name: str, model: str, api_key: str, api_base: str):
        super().__init__(name, model, api_key, api_base)

        # 内部状态变量
        self._mode = None  # "stage1_init", "stage1_chunk", "stage1_query", "stage2_label"
        self._cache_text = None  # 暂存发送的文本
        self._cache_labels = None

    # ---------------------------------------------------------------------
    # 构造 prompt 的核心逻辑（与 BaseAgent.generate 配合）
    # ---------------------------------------------------------------------
    def _build_prompt(self, **kwargs) -> str:
        """根据当前阶段构建 prompt"""
        if self._mode == "stage2_label":
            text_result = self._cache_text
            print("%%%%", len(text_result))
            label_str = ", ".join(self._cache_labels)
            # return dedent(f"""
            #     下面我将发送给你一段文本以及一个问题，请你分析文本并且回答这个问题。
            #
            #     文本如下：
            #     {text_result}
            #
            #     问题是：
            #     {label_str}
            #
            #     简洁返回答案。
            # """).strip()

            return dedent(f"""
            I will provide you a text and a question. Please answer the question **directly and concisely** using only the information in the text. 
            Do **not** start your answer with phrases like "Based on the text" or any other introductory words. 
            Do **not** provide explanations, interpretations, or extra commentary.

            Example of desired answer style:
            "Problem": "what is descriptive norm",
            "Answer": "Individual perception of others' actual behavior. For example, if most people in your office recycle paper, you might also start recycling because you perceive it as the common behavior."

            Text:
            {text_result}

            Problem:
            {label_str}

            Answer:
            """).strip()



        else:
            # stage1 部分使用 messages 直接在 process_texts 内处理
            return ""

    # ---------------------------------------------------------------------
    # 工具函数：分段
    # ---------------------------------------------------------------------
    def _split_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """将文本分段"""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # ---------------------------------------------------------------------
    # 主函数
    # ---------------------------------------------------------------------

    def process_texts(self, cause: List[str], consultation_process: List[List[str]], save_path: str = None):
        """
        处理流程：
        1️⃣ 阶段1：检测无关内容（text_result）
        2️⃣ 阶段2：预测情感标签（result）
        """
        results = []

        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        for i, (causes, consultation_process) in enumerate(zip(cause, consultation_process), start=1):
            if i < 116 or i > 122:
                continue
            print(f"\n🔹 正在处理第 {i}/{len(cause)} 条文本...")
            try:
                messages = []

                def split_long_text(text_list, max_chars=8000):
                    """将 consultation_process 按字符长度拆分为多段"""
                    chunks = []
                    temp = []
                    total_len = 0
                    for line in text_list:
                        total_len += len(line)
                        temp.append(line)
                        if total_len >= max_chars:
                            chunks.append("\n".join(temp))
                            temp = []
                            total_len = 0
                    if temp:
                        chunks.append("\n".join(temp))
                    return chunks

                # 在主循环中
                chunks = split_long_text(consultation_process, max_chars=8000)
                messages = []

                # 先逐条发送背景信息
                for chunk in chunks:
                    messages.append({"role": "user", "content": chunk})


                # 1️⃣ 分段发送 consultation_process（背景输入）
                mid = len(consultation_process) // 2
                part1 = consultation_process[:mid]
                part2 = consultation_process[mid:]

                # 直接发送第一部分
                messages.append({
                    "role": "user",
                    "content": "Consultation_process".join(part1)
                })
                # 直接发送第二部分
                messages.append({
                    "role": "user",
                    "content": "\n".join(part2)
                })

                # 2️⃣ 最后发送总结指令
                messages.append({
                    "role": "user",
                    "content": dedent(f"""
                        Causes:
                        {cause}

                        You have been provided with the full consultation process in multiple parts.
                        
                        Please help me summarize the psychotherapy case into five categories: 
                        Causes – underlying emotional, psychological, or traumatic origins of the problem. 
                        Symptoms – observable or reported psychological, behavioral, or physical manifestations. 
                        Treatment Process – step-by-step interventions and therapeutic approaches used. 
                        Characteristics of Illness – special features of the case, such as prior experience, beliefs, or complicating factors. 
                        Treatment Effect – outcomes and changes in emotions, cognition, behavior, or relationships after the intervention. 

                        Please write the summary in concise, professional English, similar to academic case reports. Keep each category clear and distinct, using one or two sentences if possible. 
                        Avoid unnecessary storytelling; focus on clinical and psychological insights.

                        Output format (strictly JSON):
                        {{"predicted_cause": "...........","predicted_symptoms": "...........","predicted_treatment_process": "...........","predicted_illness_Characteristics": "...........","predicted_treatment_effect": "..........."}}
                    """).strip()},)


                # 调用模型
                max_retry = 5  # 最大重试次数
                retry_count = 0
                data_dict = None

                while retry_count < max_retry and data_dict is None:
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages
                    )
                    text_result = response.choices[0].message.content.strip()

                    try:
                        # 尝试直接解析
                        data_dict = json.loads(text_result)
                    except json.JSONDecodeError:
                        # 清洗非标准格式
                        text_result_clean = text_result.replace("```json", "").replace("```", "").strip()
                        try:
                            data_dict = json.loads(text_result_clean)
                        except json.JSONDecodeError as e:
                            retry_count += 1
                            print(f"[Warning] 第 {retry_count} 次解析失败, 错误: {e}. 重试调用模型...")
                            # 可以选择短暂 sleep，或调整 prompt 再次请求
                            continue

                if data_dict is None:
                    print(f"[Error] 超过最大重试次数 {max_retry}，仍无法解析 JSON。")
                else:
                    # 写入文件
                    record = {"index": i, **data_dict}
                    print(record)
                    if save_path:
                        with open(save_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")


            except Exception as e:
                print(f"[Error] 第 {i} 条文本处理失败: {e}")
                results.append({
                    "irrelevant_part": None,
                    "predicted_emotion": f"[Error] {e}"
                })

        # # ========== 保存结果为 log 文件 ==========
        # if save_path:
        #     with open(save_path, "w", encoding="utf-8") as f:
        #         for idx, r in enumerate(results, start=1):
        #             f.write(f"【第{idx}条结果】\n")
        #             f.write(f"问题相关文本：{r['relevant_part']}\n")
        #             f.write(f"回答答案：{r['predicted_answer']}\n")
        #             f.write("-" * 50 + "\n")

        return results
