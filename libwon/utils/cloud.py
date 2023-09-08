import requests
def send_autodl(title="", name="exp", content="done", token = ""):
    resp = requests.post(
        "https://www.autodl.com/api/v1/wechat/message/push",
        json={
            "token": f"{token}",
            "title": f"{title}",
            "name": f"{name}",
            "content": f"{content}",
        },
    )
    return resp.content.decode()
