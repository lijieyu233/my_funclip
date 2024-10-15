#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunClip). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
import re

def time_convert(ms):
    ms = int(ms)
    tail = ms % 1000
    s = ms // 1000
    mi = s // 60
    s = s % 60
    h = mi // 60
    mi = mi % 60
    h = "00" if h == 0 else str(h)
    mi = "00" if mi == 0 else str(mi)
    s = "00" if s == 0 else str(s)
    tail = str(tail)
    if len(h) == 1: h = '0' + h
    if len(mi) == 1: mi = '0' + mi
    if len(s) == 1: s = '0' + s
    return "{}:{}:{},{}".format(h, mi, s, tail)

def str2list(text):
    pattern = re.compile(r'[\u4e00-\u9fff]|[\w-]+', re.UNICODE)
    elements = pattern.findall(text)
    return elements

class Text2SRT():
    def __init__(self, text, timestamp, offset=0):
        self.token_list = text
        self.timestamp = timestamp
        start, end = timestamp[0][0] - offset, timestamp[-1][1] - offset
        self.start_sec, self.end_sec = start, end
        self.start_time = time_convert(start)
        self.end_time = time_convert(end)
    def text(self):
        if isinstance(self.token_list, str):
            return self.token_list
        else:
            res = ""
            for word in self.token_list:
                if '\u4e00' <= word <= '\u9fff':
                    res += word
                else:
                    res += " " + word
            return res.lstrip()
    def srt(self, acc_ost=0.0):
        return "{} --> {}\n{}\n".format(
            time_convert(self.start_sec+acc_ost*1000),
            time_convert(self.end_sec+acc_ost*1000), 
            self.text())
    def time(self, acc_ost=0.0):
        return (self.start_sec/1000+acc_ost, self.end_sec/1000+acc_ost)


def generate_srt(sentence_list):
    srt_total = ''
    for i, sent in enumerate(sentence_list):
        t2s = Text2SRT(sent['text'], sent['timestamp'])
        if 'spk' in sent:
            srt_total += "{}  spk{}\n{}".format(i, sent['spk'], t2s.srt())
        else:
            srt_total += "{}\n{}".format(i, t2s.srt())
    return srt_total

def generate_srt_clip(sentence_list, start, end, begin_index=0, time_acc_ost=0.0):
    # 将开始和结束时间从秒转换为毫秒
    start, end = int(start * 1000), int(end * 1000)
    srt_total = ''  # 初始化SRT字幕文件内容为空字符串
    cc = 1 + begin_index  # 初始化当前字幕编号为起始索引加1
    subs = []  # 初始化字幕列表为空列表

    for _, sent in enumerate(sentence_list):
        # 如果句子文本是字符串类型，则转换为列表
        if isinstance(sent['text'], str):
            sent['text'] = str2list(sent['text'])

        # 如果当前句子的结束时间小于等于开始时间，则跳过该句子
        if sent['timestamp'][-1][1] <= start:
            # print("CASE0")
            continue

        # 如果当前句子的开始时间大于等于结束时间，则退出循环
        if sent['timestamp'][0][0] >= end:
            # print("CASE4")
            break

        # 处理句子时间戳在指定时间范围内的情况
        if (sent['timestamp'][-1][1] <= end and sent['timestamp'][0][0] > start) or (sent['timestamp'][-1][1] == end and sent['timestamp'][0][0] == start):
            # print("CASE1"); import pdb; pdb.set_trace()
            # 创建Text2SRT对象，并处理文本和时间戳
            t2s = Text2SRT(sent['text'], sent['timestamp'], offset=start)
            srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
            subs.append((t2s.time(time_acc_ost), t2s.text()))
            cc += 1
            continue

        # 处理句子开始时间小于等于开始时间且结束时间不大于结束时间的情况
        if sent['timestamp'][0][0] <= start:
            # print("CASE2"); import pdb; pdb.set_trace()
            if not sent['timestamp'][-1][1] > end:
                # 找到句子中第一个时间戳大于开始时间的索引
                for j, ts in enumerate(sent['timestamp']):
                    if ts[1] > start:
                        break
                _text = sent['text'][j:]
                _ts = sent['timestamp'][j:]
            else:
                # 分别找到句子中第一个时间戳大于开始时间和结束时间的索引
                for j, ts in enumerate(sent['timestamp']):
                    if ts[1] > start:
                        _start = j
                        break
                for j, ts in enumerate(sent['timestamp']):
                    if ts[1] > end:
                        _end = j
                        break
                # 截取指定时间范围内的文本和时间戳
                # _text = " ".join(sent['text'][_start:_end])
                _text = sent['text'][_start:_end]
                _ts = sent['timestamp'][_start:_end]
            if len(_ts):
                # 创建Text2SRT对象，并处理截取后的文本和时间戳
                t2s = Text2SRT(_text, _ts, offset=start)
                srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
                subs.append((t2s.time(time_acc_ost), t2s.text()))
                cc += 1
            continue

        # 处理句子结束时间大于结束时间的情况
        if sent['timestamp'][-1][1] > end:
            # print("CASE3"); import pdb; pdb.set_trace()
            # 找到句子中第一个时间戳大于结束时间的索引
            for j, ts in enumerate(sent['timestamp']):
                if ts[1] > end:
                    break
            _text = sent['text'][:j]
            _ts = sent['timestamp'][:j]
            if len(_ts):
                # 创建Text2SRT对象，并处理截取后的文本和时间戳
                t2s = Text2SRT(_text, _ts, offset=start)
                srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
                subs.append(
                    (t2s.time(time_acc_ost), t2s.text())
                    )
                cc += 1
            continue

    return srt_total, subs, cc

