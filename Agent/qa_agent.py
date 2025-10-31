# 文件名：text_agent.py
from base_agent import BaseAgent
from typing import List
import json
from textwrap import dedent

class QaAgent(BaseAgent):
    """针对文本数据逐条调用大模型的Agent（两阶段：无关段检测 + 情感标签预测）"""

    def __init__(self, name: str, model: str, api_key: str, api_base: str):
        super().__init__(name, model, api_key, api_base)

        # 内部状态变量
        self._mode = None        # "stage1_init", "stage1_chunk", "stage1_query", "stage2_label"
        self._cache_text = None  # 暂存发送的文本
        self._cache_labels = None

    # ---------------------------------------------------------------------
    # 构造 prompt 的核心逻辑（与 BaseAgent.generate 配合）
    # ---------------------------------------------------------------------
    def _build_prompt(self, **kwargs) -> str:
        """根据当前阶段构建 prompt"""
        if self._mode == "stage2_label":
            text_result = self._cache_text
            print("%%%%",len(text_result))
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
            
            For example:
            "problem": "what is descriptive norm", "answer": "Individual perception of others' actual behavior", 
             
            Text:
            {text_result}

            Question:
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
    def process_texts(self, texts: List[str], question: List[List[str]], save_path: str = None):
        """
        处理流程：
        1️⃣ 阶段1：检测无关内容（text_result）
        2️⃣ 阶段2：预测情感标签（result）
        """
        results = []

        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        for i, (text, q) in enumerate(zip(texts, question), start=1):
            print(f"\n🔹 正在处理第 {i}/{len(texts)} 条文本...")
            try:
                # 保存结果


                # ========== 阶段 1：无关内容检测 ==========
                chunks = self._split_text(text)

                # 构建 messages
                # messages = [
                #     {"role": "system", "content": "你是一位文本分析专家。"},
                #     {"role": "user", "content": dedent(f"""
                #         请一直记住核心问题：{q} ，接下来我会分几段给你发送一整份文本。
                #         请你分析总结文本中所有与问题有关的内容，并且相关内容可能会在分段发送的过程当中被分开。
                #         分段也是随机的，因此每一段的第一句与最后一句很有可能不是完整的。
                #         你需要结合整个文本去整理所有与问题有关的内容，并将这部分内容返回给我。不要给我任何的分析与解释，只给我与问题相关的内容即可
                #     """).strip()},
                # ]
                messages = [
                    {"role": "system", "content": "You are a text analysis expert."},
                    {"role": "user", "content": dedent(f"""
                        Please always keep the core question in mind: {q}. I will send you a long text in several parts. 
                        Please analyze and summarize all content related to the question. Note that the relevant content may be split across different parts.
                        The segments are sent randomly, so the first and last sentence of each part may not be complete.
                        You need to consider the entire text to organize all content related to the question and return only this content to me. 
                        Do not provide any analysis or explanation—just return the content relevant to the question.
                    """).strip()},
                ]

                # 添加所有分段内容
                for chunk in chunks:
                    messages.append({"role": "user", "content": chunk})

                # 最后追加提问
                # messages.append({"role": "user", "content": "现在请汇总并输出所有与问题有关的句子"})
                messages.append({"role": "user",
                                 "content": "Now please summarize and output all sentences related to the question."})

                # 调用模型
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages
                )
                text_result = response.choices[0].message.content.strip()
                print(f"第 {i}/{len(texts)} 条与问题相关内容为：{text_result}")

                # ========== 阶段 2：情感标签预测 ==========
                print(f"📨 开始回答问题{q}...")
                self._mode = "stage2_label"
                self._cache_text = text_result
                self._cache_labels = q
                self._build_prompt()
                result = self.generate()
                print(f"✅ 获得答案：{result}")

                # 保存结果到列表
                # results.append({
                #     "irrelevant_part": text_result,
                #     "predicted_emotion": result
                # })

                result_item = {
                    "irrelevant_part": text_result,
                    "predicted_emotion": result
                }
                results.append(result_item)

                # ✅ 立即写入文件（追加）
                if save_path:
                    with open(save_path, "a", encoding="utf-8") as f:
                        f.write(f"【第{i}个问题：{q}】\n")
                        f.write(f"问题相关文本：{result_item['irrelevant_part']}\n")
                        f.write(f"回答{i}答案：{result_item['predicted_emotion']}\n")
                        f.write("-" * 50 + "\n")
                        f.flush()

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
