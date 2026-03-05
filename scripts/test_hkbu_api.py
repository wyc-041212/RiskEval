from __future__ import annotations

import os

from openai import OpenAI


API_KEY = os.getenv("HKBUEDU_API_KEY", "9f500567-3566-48b7-8eb6-b40634cd3da1")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://genai.hkbu.edu.hk/api/v0/rest",
)


def main() -> None:
    print("🔍 測試 models.list() ...")
    try:
        models = client.models.list()
        print("✅ 成功！拿到 models 數量 =", len(models.data))
        for m in models.data:
            print(" -", m.id)
    except Exception as e:
        print("❌ 呼叫失敗，錯誤如下：")
        print(e)


if __name__ == "__main__":
    main()
