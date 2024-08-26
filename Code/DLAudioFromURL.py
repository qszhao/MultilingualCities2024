import re
import requests
# read the audio from the link

# myString = "This is a link http://www.google.com"
# print(re.search("(?P<url>https?://[^\s]+)", myString).group("url"))

with open("D:\\dataset\\Gaelic\\url.txt", "r", encoding='utf-8') as f:  # 打开文件
    data = f.read()  # 读取文件

urlall = re.findall(r'(https?://[^\s]+.mp3)', data)

urlall = urlall
# generate the audio set
len = len(urlall)
for i in range(783,len):
    res = requests.get(urlall[i])
    music = res.content
    with open(r'D:/dataset/Gaelic/%d.mp3' % (i+1), 'ab') as file: #保存到本地的文件名
        file.write(res.content)
        file.flush()
    print(i)
