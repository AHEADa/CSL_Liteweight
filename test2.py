
import re
import os
import sys

from pydub import AudioSegment
from pydub.playback import play
from xpinyin import Pinyin


def chinese_phonetic(wenzi_data):
    """
    :param wenzi_data: data
    :return: 拼音
    """
    p = Pinyin()
    punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{} \n'
    wenzi_data = re.sub(r"[%s]+" % punc, "0", wenzi_data)
    ret = p.get_pinyin(wenzi_data, tone_marks='numbers')
    voi = ret.split('-')
    return voi


def speed_change(sound, speed=1.2):
    """
    :param sound:
    :param speed: 音频速度
    :return:
    """

    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
         "frame_rate": int(sound.frame_rate * speed)
      })

    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)


def Sound_Size(sound):
    """
    :param sound:
    :return: 调整声音大小
    """
    sound = sound + 15
    return sound

def save_five(sound):
    """
    :param sound:
    :return: mp3
    """
    sound.export(os.path.join('vvv.wav'), format="wav")


def main_voice(wenzi_data):
    voi = chinese_phonetic(wenzi_data)
    sounds = []
    playlist = AudioSegment.empty()
    for x, i in enumerate(voi):
        i = i.lower()
        try:
            if i == '0':
                sounds.append(AudioSegment.silent(duration=230, frame_rate=1600))
            sounds.append(AudioSegment.from_wav(r"C:\\Users\\AHEAD\\Desktop\\科创手语翻译\\test2\\voice\\" + i + ".wav"))
        except:
            # sounds.append(AudioSegment.silent(duration=10000, frame_rate=16000))
            print(i)
    for sound in sounds:
        playlist += sound
    playlist = speed_change(playlist)
    playlist = Sound_Size(playlist)
    #play(playlist)
    save_five(playlist)# 保存


p = """你好，很高兴认识你。
"""
main_voice(p)
