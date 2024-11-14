from openai import OpenAI

client = OpenAI(
    base_url="http://0.0.0.0:8001/v1",  # Changed to port 8001
    api_key="dummy-key"
)

response = client.chat.completions.create(
    model="/mnt/chatbot30TB/junghyunlee/juneyang/Qwen/Qwen2.5-Math-7B-Instruct",
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)

print(response.choices[0].message.content)