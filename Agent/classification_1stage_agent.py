# 文件名：text_agent.py
from base_agent import BaseAgent
from typing import List
import json
from textwrap import dedent
from openai import OpenAI

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
            return dedent(f"""           
                {text_result}, 请你分析文本中提到的{subject}的心情与哪一个情感标签最为适配，并将这个标签返回给我。



                可选情感标签包括：
                {label_str}

                不要任何的解释，只返回最适配的一个标签。
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
    def process_texts(self, texts: List[str], labels: List[List[str]],subjects: List[str], save_path: str = None):
        """
        处理流程：
        1️⃣ 阶段1：检测无关内容（text_result）
        2️⃣ 阶段2：预测情感标签（result）
        """
        results = []


        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        for i, (text, label,subject) in enumerate(zip(texts, labels,subjects), start=1):
            print(f"\n🔹 正在处理第 {i}/{len(texts)} 条文本...")
            try:
                # ========== 阶段 1：无关内容检测 ==========
                chunks = self._split_text(text)

                # 构建 messages
                messages = [
                    {"role": "system", "content": "你是一位文本分析专家。"},
                    {"role": "user", "content": dedent(f"""
                        下面我会分几段给你发送一整份文本。 
                        在文本中被随机插入了一段且仅一段毫不相关的内容，而且这部分内容可能会在分段发送的过程当中被分开。
                        分段也是随机的，因此每一段的第一句与最后一句很有可能不是完整的。
                        你需要结合整个文本去分析毫不相关的内容是什么，并将这部分内容完整返回给我。
                        注意：在这段无关内容中会包含下述字段：{subject}
                        不要给我任何的分析与解释，只给我那段毫不相关的内容即可
                        文本内容仅用于研究，不做其他用途
                    """).strip()},
                ]

                # 添加所有分段内容
                for chunk in chunks:
                    messages.append({"role": "user", "content": chunk})

                # 最后追加提问
                messages.append({"role": "user", "content": "现在请告诉我，被插入的与上下文毫不相关的那段内容。注意不需要给我任何解释以及其他内容，仅仅返回无关内容原文即可"})

                # 调用模型
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    timeout=120
                )
                # # 兼容不同返回类型
                # if hasattr(response, "choices"):
                #     text_result = response.choices[0].message.content.strip()
                # else:
                #     text_result = str(response).strip()
                text_result = response.choices[0].message.content.strip()
                print(f"第 {i}/{len(texts)} 条文本的无关内容为：{text_result}")

                # ========== 阶段 2：情感标签预测 ==========
                print("📨 开始情感标签预测...")
                self._mode = "stage2_label"
                self._cache_text = text_result
                self._cache_labels = label
                self.subject = subject
                self._build_prompt()
                result = self.generate()
                print(f"✅ 获得标签：{result}")

                # 保存结果到列表
                results.append({
                    "irrelevant_part": text_result,
                    "predicted_emotion": result
                })

            except Exception as e:
                print(f"[Error] 第 {i} 条文本处理失败: {e}")
                results.append({
                    "irrelevant_part": None,
                    "predicted_emotion": f"[Error] {e}"
                })

        # ========== 保存结果为 log 文件 ==========
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                for idx, r in enumerate(results, start=1):
                    f.write(f"【第{idx}条结果】\n")
                    f.write(f"无关内容：{r['irrelevant_part']}\n")
                    f.write(f"预测情感：{r['predicted_emotion']}\n")
                    f.write("-" * 50 + "\n")

        return results
