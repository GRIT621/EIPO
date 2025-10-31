# 文件名：text_agent.py
from base_agent import BaseAgent
from typing import List
import json
from textwrap import dedent

class TextAgent(BaseAgent):
    """针对文本数据逐条调用大模型的Agent（两阶段：无关段检测 + 情感标签预测）"""

    def __init__(self, name: str, model: str, api_key: str, api_base: str):
        super().__init__(name, model, api_key, api_base)

        # 内部状态变量
        self._mode = None        # "stage1_init", "stage1_chunk", "stage1_query", "stage2_label"
        self._cache_text = None  # 暂存发送的文本
        self._cache_labels = None
        self.subject = None

    # ---------------------------------------------------------------------
    # 构造 prompt 的核心逻辑（与 BaseAgent.generate 配合）
    # ---------------------------------------------------------------------
    def _build_prompt(self, **kwargs) -> str:
        """根据当前阶段构建 prompt"""
        if self._mode == "stage2_label":
            text_result = self._cache_text
            subject = self.subject
            print("subject:",subject)
            print("****",len(self._cache_text))
            label_str = ", ".join(self._cache_labels)
            # return dedent(f"""
            #     **Instructions**
            #     In this task, you are presented with a scenario, a question, and multiple choices. Please carefully analyze
            #     the scenario and take the perspective of the individual involved.
            #     **Note**
            #     Provide only one single correct answer to the question and respond only with the words in choices
            #     Do not provide explanations for your response.
            #     Scenario:{text_result}
            #     Question: What emotion(s) would {subject} ultimately feel in this situation?
            #     Choices:{label_str}
            # """).strip()
            return dedent(f"""
            {text_result}. {label_str}.
            According to the *current* situation, what do you think *{subject}*'s emotions are *right now*?
            
            !Pay attention to the subject being analyzed. 

            #Provide only one single correct answer to the question and respond only with the words in choices.
            #Do not provide explanations for your response.
            """)
            # return dedent(f"""
            # **Task Instructions**
            # You are presented with a short scenario and a question about the emotional state of a specific person.
            # Your goal is to *understand* the person's feelings — not by matching obvious words, but by reasoning about
            # their perspective, beliefs, and the emotional implications of the events.

            # **Reasoning Steps**
            # 1. Read the scenario carefully and identify what happened.
            # 2. Consider how the event would be *personally meaningful* to {subject}.
            # 3. Think about what the person expected, what actually happened, and how this mismatch affects their feelings.
            # 4. Based on this reasoning, determine which emotion(s) best describe {subject}'s *final emotional state*.

            # **Scenario:** {text_result}

            # **Question:** What emotion(s) would {subject} ultimately feel in this situation?

            # **Choices:** {label_str}

            # **Output Requirement**
            # Respond with **only one choice** from the list above.
            # Do not include explanations or reasoning in your final answer.
            # """).strip()

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
    def process_texts(self, texts: List[str], labels: List[List[str]],subjects: List[str], save_path: str = None):
        """
        处理流程：
        1️⃣ 阶段1：检测无关内容（text_result）
        2️⃣ 阶段2：预测情感标签（result）
        """
        results = []

        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        for i, (text, label,subject) in enumerate(zip(texts, labels,subjects), start=1):
            print(f"\n🔹 正在处理第 {i}/{len(texts)} 条文本...")
            try:
                
                # ========== 阶段 2：情感标签预测 ==========
                print("📨 开始情感标签预测...")
                self._mode = "stage2_label"
                self._cache_text = text
                self._cache_labels = label
                self.subject = subject
                self._build_prompt()
                result = self.generate()
                print(f"✅ 获得标签：{result}")

                # 保存结果到列表
                results.append({
                    # "irrelevant_part": text_result,
                    "predicted_emotion": result
                })

            except Exception as e:
                print(f"[Error] 第 {i} 条文本处理失败: {e}")
                results.append({
                    # "irrelevant_part": None,
                    "predicted_emotion": f"[Error] {e}"
                })

        # # ========== 保存结果为 log 文件 ==========
        # if save_path:
        #     with open(save_path, "w", encoding="utf-8") as f:
        #         for idx, r in enumerate(results, start=1):
        #             f.write(f"【第{idx}条结果】\n")
        #             f.write(f"无关内容：{r['irrelevant_part']}\n")
        #             f.write(f"预测情感：{r['predicted_emotion']}\n")
        #             f.write("-" * 50 + "\n")
        # ========== 保存结果为 JSONL 文件 ==========
                # ========== 保存结果为 JSONL 文件 ==========
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                for idx, (r, text, subject) in enumerate(zip(results, texts, subjects)):
                    record = {
                        "id": idx,  # 从0开始编号
                        "predicted_emotion": r["predicted_emotion"]
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return results
