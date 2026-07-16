import requests

response = requests.get("https://jsonplaceholder.typicode.com/posts/1", verify=False)

print(response.status_code)

if response.status_code == 200:
    print(f"内容: {response.text}")
    print(response.json().get("url"))
else:
    print("请求失败")
