#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunClip). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import re
import os
import sys
import copy
import librosa
import logging
import argparse
import numpy as np
import soundfile as sf
from moviepy.editor import *
import moviepy.editor as mpy
from moviepy.video.tools.subtitles import SubtitlesClip, TextClip
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.compositing import CompositeVideoClip

from utils.subtitle_utils import generate_srt, generate_srt_clip
from utils.argparse_tools import ArgumentParser, get_commandline_args
from utils.trans_utils import pre_proc, proc, write_state, load_state, proc_spk, convert_pcm_to_float

logger=logging.getLogger('my_logger')
class VideoClipper():
    def __init__(self, funasr_model):
        logging.info("初始化 VideoClipper.")
        self.funasr_model = funasr_model # 传入的funasr_model对象
        self.GLOBAL_COUNT = 0

    # 识别视频 生成音频字幕和文字
    # 参数：
    # video_filename: 视频文件路径
    # sd_switch: 是否开启语义理解
    # hotwords: 热词列表
    # output_dir: 输出目录，默认为None
    # 返回值：
    # 识别后的文本、字幕文件路径

    #todo 音频识别
    def video_recog(self, video_filename, sd_switch='no', hotwords="", output_dir=None):
        logger.info("video_recog:提取视频音频进行下一步处理")
        # 读取视频文件
        video = mpy.VideoFileClip(video_filename)  #读取视频
        # Extract the base name, add '_clip.mp4', and 'wav'

        # 处理文件路径,创建根据原视频文件创建切片文件名
        logger.warning("处理文件路径 根据原视频文件创建切片文件名")
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            _, base_name = os.path.split(video_filename) # 分离掉临时文件的路径
            base_name, _ = os.path.splitext(base_name)   # 分离掉扩展名
            clip_video_file = base_name + '_clip.mp4'    # 加上新后缀名

            audio_file = base_name + '.wav'
            audio_file = os.path.join(output_dir, audio_file)

        else:
            # output_dir为空 保存到临时文件中 经过我的修改 output不可能为空
            base_name, _ = os.path.splitext(video_filename) # 分离掉扩展名 #相当于保存在输入视频的路径 也就是临时文件路径
            clip_video_file = base_name + '_clip.mp4'  # 加上新后缀
            audio_file = base_name + '.wav'

        # 读取视频中的音频 临时保存到音频文件中
        video.audio.write_audiofile(audio_file)

        # 识别音频后删除音频文件
        logger.info("获取音频 加载的音频将被重采样到16 kHz")
        wav = librosa.load(audio_file, sr=16000)[0]
        if os.path.exists(audio_file):
            os.remove(audio_file)
        state = {
            'video_filename': video_filename, # 原视频文件路径
            'clip_video_file': clip_video_file, # 剪辑后的视频文件路径
            'video': video,  # VideoFileClip视频对象
        }
        logger.info(f"当前state:{state}")

        logging.warning("开始识别音频,生成文字")
        print("videoclipper video_recog state:"+str(state))
        # res_text, res_srt = self.recog((16000, wav), state)
        return self.recog((16000, wav), sd_switch, state, hotwords, output_dir)

    # 识别音频
    # 参数：
    # audio_input: 从视频上获取的音频，元组形式的音频数据，包括采样率和数据
    # sd_switch: 是否开启语义理解 默认关闭
    # state: {
    #       'video_filename': video_filename, # 原视频文件路径
    #       'clip_video_file': clip_video_file, # 剪辑后的视频文件路径
    #       'video': video,  # VideoFileClip视频对象
    # }
    # hotwords: 热词列表
    # output_dir: 输出目录，默认为None
    # 返回值：
    # 识别后的文本、字幕文件路径
    def recog(self, audio_input, sd_switch='no', state=None, hotwords="", output_dir=None):
        if state is None:
            state = {}
        sr, data = audio_input # 获取音频采样率和数据

        # Convert to float64 consistently (includes data type checking)
        # 转编码之类
        data = convert_pcm_to_float(data)
        # assert sr == 16000, "16kHz sample rate required, {} given.".format(sr)
        if sr != 16000: # resample with librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        if len(data.shape) == 2:  # multi-channel wav input
            logging.warning("Input wav shape: {}, only first channel reserved.".format(data.shape))
            data = data[:,0]
        logger.info("state中添加 音频数据和音频采样率")
        state['audio_input'] = (sr, data)

        # 判断是否开启语义理解 默认关闭
        if sd_switch == 'Yes':
            rec_result = self.funasr_model.generate(data,
                                                    return_spk_res=True,
                                                    return_raw_text=True,
                                                    is_final=True,
                                                    output_dir=output_dir,
                                                    hotword=hotwords,
                                                    pred_timestamp=self.lang=='en',
                                                    en_post_proc=self.lang=='en',
                                                    cache={})
            res_srt = generate_srt(rec_result[0]['sentence_info'])
            state['sd_sentences'] = rec_result[0]['sentence_info']

        # 未开启语义理解 调用funasr模型 识别字幕 输出到指定目录
        else:
            rec_result = self.funasr_model.generate(data,# 音频数据
                                                    return_spk_res=False, # 是否返回语音识别结果
                                                    sentence_timestamp=True, # 是否返回时间戳
                                                    return_raw_text=True, # 是否返回原始文本
                                                    is_final=True, # 是否返回最终结果
                                                    hotword=hotwords, # 热词列表
                                                    output_dir=output_dir, # 输出目录
                                                    pred_timestamp=self.lang=='en', # 是否预测时间戳
                                                    en_post_proc=self.lang=='en', # 是否进行英文后处理
                                                    cache={}) # 缓存信息
            logger.info(f"调用funasrt生成rec_result{rec_result}")
            # 生成srt 字幕文件
            logger.info("根据rec_result生成srt 字幕文件")
            res_srt = generate_srt(rec_result[0]['sentence_info'])

        # 添加state信息
        state['recog_res_raw'] = rec_result[0]['raw_text']
        state['timestamp'] = rec_result[0]['timestamp']
        state['sentences'] = rec_result[0]['sentence_info']
        logging.info(f"state:{state}")
        res_text = rec_result[0]['text']
        return res_text, res_srt, state





    # 根据时间戳列表剪辑视频
    def my_video_clip(self,
                   dest_text,  # 原始文本
                   start_ost,  # 视频开始偏移量（毫秒）
                   end_ost,  # 视频结束偏移量（毫秒）
                   state,  # 存储识别结果的状态对象
                   font_size=32,  # 字幕的字体大小
                   font_color='white',  # 字幕的字体颜色
                   add_sub=False,  # 是否添加字幕，默认不添加
                   dest_spk=None,  # 目标说话人，用于匹配特定说话人的时间段
                   output_dir=None,  # 输出文件目录
                   timestamp_list=None):  # AI生成字幕的时间戳

        # 从 state 中提取识别的原始结果、时间戳、句子和视频信息
        sentences = state['sentences'] # 包含句子和对应时间戳的列表，每个句子对应一个时间段 在llm剪辑中使用
        video = state['video'] # 当前处理的视频对象，moviepy 的 VideoFileClip 对象，用于截取和操作视频片段
        clip_video_file = state['clip_video_file'] # 原始文件名+_clip.mp4        剪辑后生成的视频文件名和路径，用于保存最终剪辑
        video_filename = state['video_filename'] # 原始文件路径 输入视频文件的名称，代表正在处理的原始视频 video_filename = "input_video.mp4"
        logger.info("videoclipper:::my_video_clip::clip_video_file:" + clip_video_file)
        logger.info("videoclipper:::my_video_clip::video_filename:" + video_filename)


        # 获取时间戳
        all_ts = [[i[0] * 16.0, i[1] * 16.0] for i in timestamp_list] # 将时间戳列表转换为帧数，每个时间段对应一个帧数
        srt_index = 0  # 字幕的索引，用于逐条处理字幕 编号与切分的视频数统一
        time_acc_ost = 0.0  # 累积的时间偏移，用于处理多个片段
        ts = all_ts  # 保存所有时间戳
        clip_srt = ""  # 保存生成的 SRT 字幕内容



        # 如果有匹配的时间戳，则进行视频片段的剪辑
        if len(ts): # 如果时间戳列表不为空
            if self.lang == 'en' and isinstance(sentences, str): # 如果字幕是英文且句子是字符串类型，按空格拆分为单词
                sentences = sentences.split()


            start, end = ts[0][0] / 16000, ts[0][1] / 16000 # 获取第一个时间段的起始和结束时间，并将帧数转换为秒
            srt_clip, subs, srt_index = generate_srt_clip(sentences, start, end, begin_index=srt_index, # 生成字幕片段 通过start end裁剪该视频时间端的字幕
                                                          time_acc_ost=time_acc_ost)
            start, end = start + start_ost / 1000.0, end + end_ost / 1000.0 # 应用偏移时间，调整片段的起止时间


            # 从视频中截取这个片段 加入列表
            video_clip = video.subclip(start, end)
            start_end_info = "from {} to {}".format(start, end)  # 记录开始和结束信息
            clip_srt += srt_clip  # 添加字幕
            concate_clip = [video_clip]  # 视频片段列表，用于存储所有片段
            time_acc_ost += end + end_ost / 1000.0 - (start + start_ost / 1000.0)  # 累积时间偏移

            # 处理剩下的时间戳，并生成更多视频片段
            for _ts in ts[1:]:  # 从第二个时间戳开始处理
                start, end = _ts[0] / 16000, _ts[1] / 16000 # 获取当前时间段的开始和结束时间
                srt_clip, subs, srt_index = generate_srt_clip(sentences, start, end, begin_index=srt_index - 1,# 生成字幕片段
                                                              time_acc_ost=time_acc_ost)
                # 如果没有字幕，则跳过这个片段
                if not len(subs):
                    continue
                chi_subs = []  # 存储调整后的字幕
                sub_starts = subs[0][0][0]  # 第一个字幕的开始时间

                # 遍历字幕并将其时间调整为相对于当前片段的时间
                for sub in subs:
                    # 将字幕的时间减去片段的起始时间，得到相对时间 用于给切片加字幕
                    chi_subs.append(((sub[0][0] - sub_starts, sub[0][1] - sub_starts), sub[1]))

                # 应用时间偏移，调整片段的开始和结束时间
                start, end = start + start_ost / 1000.0, end + end_ost / 1000.0
                # 截取视频片段
                _video_clip = video.subclip(start, end)
                # 记录开始和结束时间信息
                start_end_info += ", from {} to {}".format(str(start)[:5], str(end)[:5])
                clip_srt += srt_clip  # 将字幕拼接起来
                # 将生成的片段复制并添加到片段列表中
                concate_clip.append(copy.copy(_video_clip))
                # 累积时间偏移，用于处理下一个片段
                time_acc_ost += end + end_ost / 1000.0 - (start + start_ost / 1000.0)

            # 输出日志信息，记录处理了多少时间段
            message = "{} periods found in the audio: ".format(len(ts)) + start_end_info
            logger.warning("Concating...")

            # 如果指定了输出目录（output_dir 不为空）
            for single_clip in concate_clip:
                logger.info(f"切片生成 output_dir={output_dir}")
                os.makedirs(output_dir, exist_ok=True)
                # 分离剪辑视频文件的路径和文件名（包括扩展名） 去除路径不要
                _, file_with_extension = os.path.split(clip_video_file)
                # 将文件名与扩展名分开，获取不含扩展名的文件名 去除后缀 如.mp4
                clip_video_file_name, _ = os.path.splitext(file_with_extension)
                logger.info(f'output_dir:{output_dir}\n clip_video_file:{clip_video_file}')
                # 在输出目录中生成新的剪辑文件名，格式为 "原文件名_noX.mp4"，其中 X 是全局计数器 GLOBAL_COUNT
                new_clip_video_file = os.path.join(output_dir,
                                               "{}_no{}.mp4".format(clip_video_file_name, self.GLOBAL_COUNT))
                # 生成临时音频文件的路径，格式为 "原文件名_tempaudio_noX.mp4"
                temp_audio_file = os.path.join(output_dir,
                                               "{}_tempaudio_no{}.mp4".format(clip_video_file_name, self.GLOBAL_COUNT))

                single_clip.write_videofile(new_clip_video_file, audio_codec="aac", temp_audiofile=temp_audio_file)
                logger.info("切片生成路径:"+new_clip_video_file)
                self.GLOBAL_COUNT += 1  # 更新全局计数器，用于处理多个文件


        else:
            # 如果没有找到合适的片段，则返回原始视频文件
            clip_video_file = video_filename
            message = "No period found in the audio, return raw speech. You may check the recognition result and try other destination text."
            srt_clip = ''

        # 返回生成的视频文件、日志信息和 SRT 字幕内容
        return clip_video_file, message, clip_srt

































def get_parser():
    parser = ArgumentParser(
        description="ClipVideo Argument",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=(1, 2),
        help="Stage, 0 for recognizing and 1 for clipping",
        required=True
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Input file path",
        required=True
    )
    parser.add_argument(
        "--sd_switch",
        type=str,
        choices=("no", "yes"),
        default="no",
        help="Turn on the speaker diarization or not",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./output',
        help="Output files path",
    )
    parser.add_argument(
        "--dest_text",
        type=str,
        default=None,
        help="Destination text string for clipping",
    )
    parser.add_argument(
        "--dest_spk",
        type=str,
        default=None,
        help="Destination spk id for clipping",
    )
    parser.add_argument(
        "--start_ost",
        type=int,
        default=0,
        help="Offset time in ms at beginning for clipping"
    )
    parser.add_argument(
        "--end_ost",
        type=int,
        default=0,
        help="Offset time in ms at ending for clipping"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default='zh',
        help="language"
    )
    return parser


def runner(stage, file, sd_switch, output_dir, dest_text, dest_spk, start_ost, end_ost, output_file, config=None, lang='zh'):
    audio_suffixs = ['.wav','.mp3','.aac','.m4a','.flac']
    video_suffixs = ['.mp4','.avi','.mkv','.flv','.mov','.webm','.ts','.mpeg']
    _,ext = os.path.splitext(file)
    if ext.lower() in audio_suffixs:
        mode = 'audio'
    elif ext.lower() in video_suffixs:
        mode = 'video'
    else:
        logging.error("Unsupported file format: {}\n\nplease choise one of the following: {}".format(file),audio_suffixs+video_suffixs)
        sys.exit(1) # exit if the file is not supported
    while output_dir.endswith('/'):
        output_dir = output_dir[:-1]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if stage == 1:
        from funasr import AutoModel
        # initialize funasr automodel
        logging.warning("Initializing modelscope asr pipeline.")
        if lang == 'zh':
            funasr_model = AutoModel(model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                    vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                    punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                    spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                    )
            audio_clipper = VideoClipper(funasr_model)
            audio_clipper.lang = 'zh'
        elif lang == 'en':
            funasr_model = AutoModel(model="iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
                                vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                                punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                                spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                                )
            audio_clipper = VideoClipper(funasr_model)
            audio_clipper.lang = 'en'
        if mode == 'audio':
            logging.warning("Recognizing audio file: {}".format(file))
            wav, sr = librosa.load(file, sr=16000)
            res_text, res_srt, state = audio_clipper.recog((sr, wav), sd_switch)
        if mode == 'video':
            logging.warning("Recognizing video file: {}".format(file))
            res_text, res_srt, state = audio_clipper.video_recog(file, sd_switch)
        total_srt_file = output_dir + '/total.srt'
        with open(total_srt_file, 'w') as fout:
            fout.write(res_srt)
            logging.warning("Write total subtitle to {}".format(total_srt_file))
        write_state(output_dir, state)
        logging.warning("Recognition successed. You can copy the text segment from below and use stage 2.")
        print(res_text)
    if stage == 2:
        audio_clipper = VideoClipper(None)
        if mode == 'audio':
            state = load_state(output_dir)
            wav, sr = librosa.load(file, sr=16000)
            state['audio_input'] = (sr, wav)
            (sr, audio), message, srt_clip = audio_clipper.clip(dest_text, start_ost, end_ost, state, dest_spk=dest_spk)
            if output_file is None:
                output_file = output_dir + '/result.wav'
            clip_srt_file = output_file[:-3] + 'srt'
            logging.warning(message)
            sf.write(output_file, audio, 16000)
            assert output_file.endswith('.wav'), "output_file must ends with '.wav'"
            logging.warning("Save clipped wav file to {}".format(output_file))
            with open(clip_srt_file, 'w') as fout:
                fout.write(srt_clip)
                logging.warning("Write clipped subtitle to {}".format(clip_srt_file))
        if mode == 'video':
            state = load_state(output_dir)
            state['video_filename'] = file
            if output_file is None:
                state['clip_video_file'] = file[:-4] + '_clip.mp4'
            else:
                state['clip_video_file'] = output_file
            clip_srt_file = state['clip_video_file'][:-3] + 'srt'
            state['video'] = mpy.VideoFileClip(file)
            clip_video_file, message, srt_clip = audio_clipper.video_clip(dest_text, start_ost, end_ost, state, dest_spk=dest_spk)
            logging.warning("Clipping Log: {}".format(message))
            logging.warning("Save clipped mp4 file to {}".format(clip_video_file))
            with open(clip_srt_file, 'w') as fout:
                fout.write(srt_clip)
                logging.warning("Write clipped subtitle to {}".format(clip_srt_file))


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    runner(**kwargs)


if __name__ == '__main__':
    main()