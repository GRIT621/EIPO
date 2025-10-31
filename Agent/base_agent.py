from openai import OpenAI

class BaseAgent:
    def __init__(self, name: str, model: str, api_key: str, api_base: str):
        self.name = name
        self.model_name = model
        self.api_key = api_key
        self.api_base = api_base or "https://api.openai.com/v1"


    def _build_prompt(self, *args, **kwargs) -> str:
        pass

    def _call_model(self, prompt: str) -> str:
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        # reasoning_log = response.choices[0].message.reasoning_content
        # # 写入日志文件
        # with open("/Users/grit/PycharmProjects/LongEmo/result/deepseek_reasoning_3point_log.txt", "a", encoding="utf-8") as f:
        #     f.write(reasoning_log + "\n" + "=" * 80 + "\n")
        #
        # print("已成功存入日志")
        return response.choices[0].message.content

    def generate(self, *args, **kwargs):
        prompt = self._build_prompt(*args, **kwargs)
        print("@@@@@",len(prompt))
        return self._call_model(prompt)
