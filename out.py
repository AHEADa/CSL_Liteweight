import win32com.client
from Pinyin2Hanzi import DefaultHmmParams
from Pinyin2Hanzi import viterbi

# 隐马尔可夫法
hmmparams = DefaultHmmParams()

speaker = win32com.client.Dispatch("SAPI.SpVoice")
str1 = """你好，很高兴认识你。"""
#speaker.Speak(str1)

def speak(sentance: str) -> None:
    speaker.Speak(sentance)

def toText(pinyin: list) -> str:
    resultStr = []
    result = viterbi(hmm_params = hmmparams, observations = pinyin, path_num = 1)
    for item in result[0].path:
        #print(item.path)
        resultStr.append(item)

    return ''.join(resultStr)

def voice(pinyin: list) -> None:
    if pinyin != []:
        try:
            speak(toText(pinyin))
        except:
            pass

### 测试 ##
#voice(['ni','chi','le','ma'])
