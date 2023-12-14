import pyttsx3

engine = pyttsx3.init()

# 设置新的语音速率
engine.setProperty('rate', 150)

# 设置新的语音音量，音量最小为 0，最大为 1
engine.setProperty('volume', 1.0)

# 获取当前语音声音的详细信息
voices = engine.getProperty('voices')
print(f'语音声音详细信息：{voices}')
# 设置当前语音声音为女性，当前声音不能读中文
#engine.setProperty('voice', voices[1].id)
# 设置当前语音声音为男性，当前声音可以读中文
engine.setProperty('voice', voices[0].id)
# 获取当前语音声音
voice = engine.getProperty('voice')
print(f'语音声音：{voice}')
# 语音文本
words = """你好，很高兴认识你。"""
# 将语音文本说出来
engine.say(words)
engine.runAndWait()
engine.stop()