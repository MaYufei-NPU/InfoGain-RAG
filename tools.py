from openai import OpenAI
import os
import requests
import json
import requests


def aliyun_inference(model_name,messages):
    os.environ['DASHSCOPE_API_KEY'] = 'xxxxxxxxxxxxxxxxxxxxx'
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
    )
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    return completion.choices[0].message.content


def openai_inference(model_name,messages):

    client = OpenAI(
        api_key="xxxxxxxxxxxxxxxxxxxxx",  # This is the default and can be omitted
        base_url = "https://api.feidaapi.com/v1"
    )
    
    completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
    )
    
    return completion.choices[0].message.content

def llama_inference(model_name,messages):

    client = OpenAI(
        api_key="xxxxxxxxxxxxxxxxxxxxx",  # This is the default and can be omitted
        base_url = "https://api.llama-api.com"
    )
    
    completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
    )

    return completion.choices[0].message.content

def custom_inference(model_name,messages):
    # print(messages)
    response = requests.post(
        url="https://api.aimlapi.com/chat/completions",
        headers={
            "Authorization": "xxxxxxxxxxxxxxxxxxxxx",
            "Content-Type": "application/json",
        },
        data=json.dumps(
            {
                "model": model_name,
                "messages": messages,
                "max_tokens": 200,
                "stream": False,
            }
        ),
    )
    # print(response.json())
    return response.json()['choices'][0]['message']['content']

def deepseek_inference(model_name,messages):

    client = OpenAI(api_key="xxxxxxxxxxxxxxxxxxxxx", base_url="https://api.deepseek.com")
    
    completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
        stream=False
    )

    return completion.choices[0].message.content


def deepseek_r1_inference(model_name,messages):
    # print(messages)
    openai = OpenAI(
        api_key="xxxxxxxxxxxxxxxxxxxxx",
        base_url="https://api.deepinfra.com/v1/openai",
    )

    chat_completion = openai.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    # print(chat_completion)
    # print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content