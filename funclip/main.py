import os
import sys

from funclip.llm.openai_api import my_openai_call
from funclip.utils.logger_setup import setup_logger
from funclip.utils.trans_utils import extract_timestamps
from videoclipper import VideoClipper
import argparse
from funasr import AutoModel

logger = setup_logger()

# 模型加载
parser = argparse.ArgumentParser(description='argparse testing')
parser.add_argument('--lang', '-l', type=str, default="zh", help="language")
parser.add_argument('--share', '-s', action='store_true', help="if to establish gradio share link")
parser.add_argument('--port', '-p', type=int, default=7860, help='port number')
parser.add_argument('--listen', action='store_true', help="if to listen to all hosts")
args = parser.parse_args()
# 加载语音识别模型
if args.lang == 'zh':
    funasr_model = AutoModel(model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                             vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                             punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                             spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                             )
else:
    funasr_model = AutoModel(model="iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
                             vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                             punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                             spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                             )
audio_clipper = VideoClipper(funasr_model)
audio_clipper.lang = args.lang  # 设置语言


# 全局变量
# video_state = None
# audio_state = None


# todo 1.提取音频 生成字幕
# 从视频中提取音频处理
def video_recog(video_input, sd_switch, hotwords, output_dir):  # 从视频中提取音频处理
    return audio_clipper.video_recog(video_input, sd_switch, hotwords, output_dir=output_dir)


# 提取音频 或从视频中提取音频生成srt
def mix_recog(video_input, audio_input, hotwords, output_dir='../asset/切片目录'):
    # 输出路径处理
    output_dir = output_dir.strip()  # 清理输出路径中的多余空白
    logger.info('mix_recog:用户指定了输出路径,输出路径为:' + output_dir)
    output_dir = os.path.abspath(output_dir)  # 将相对路径转换为绝对路径
    audio_state, video_state = None, None  # 初始化状态

    # 优先调用视频
    if video_input is not None:
        res_text, res_srt, video_state = video_recog(
            video_input, 'No', hotwords, output_dir=output_dir)  # 调用视频识别
        # todo 此地生成的输出路径output_dir 保存到全局变量
        return res_text, res_srt, video_state, None, output_dir  # 返回视频识别结果和状态


# todo 2. AI推理 划分字幕
def llm_inference(system_content, user_content, srt_text, model, apikey="sk-MIBD51NOevuLr38Pj9XRmS7xMrm8oprTIyv312FNscHSjcpo"):
    SUPPORT_LLM_PREFIX = ['qwen', 'gpt', 'g4f', 'moonshot']
    if model.startswith('gpt') or model.startswith('moonshot'):
        return my_openai_call(apikey, model, system_content, user_content + '\n' + srt_text)
    else:
        logger.error("LLM name error, only {} are supported as LLM name prefix."
                     .format(SUPPORT_LLM_PREFIX))

# todo 3.根据AI生成后的字幕 划分视频 进行剪辑
def AI_clip(LLM_res, dest_text, video_spk_input, start_ost, end_ost, video_state,output_dir):  #
    timestamp_list = extract_timestamps(LLM_res)  # 提取生成后字幕的时间戳
    logger.info("timestamp_list:{}".format(timestamp_list))
    output_dir = output_dir.strip()  # 移除路径中的空格
    if not len(output_dir):  # 如果路径为空，将 output_dir 设置为 None
        output_dir = None  # 如果路径为空，将 output_dir 设置为 None
    else:
        output_dir = os.path.abspath(output_dir)  # 转换为绝对路径

    if video_state is not None:  # 如果视频状态不为空
        clip_video_file, message, clip_srt = audio_clipper.my_video_clip(  # 调用自定义的视频剪辑函数
            dest_text, # 原始字幕
            start_ost,
            end_ost,
            video_state,  # 传入参数
            dest_spk=video_spk_input, # 按说话人剪辑
            output_dir=output_dir,   # 输出目录
            timestamp_list=timestamp_list, #AI字幕时间戳
            add_sub=False)  # 是否添加字幕
        return clip_video_file, None, message, clip_srt  # clip_srt 返回的字幕文件


def main():
    video_input = '../asset/测试视频/test.mp4'  # 输入视频路径
    hotwords = ''  # 热词
    output_dir = '../asset/测试视频/output'  # 输出路径

    res_text = ''  # 识别文本结果
    res_srt = ''  # 识别srt结果
    video_state = None  # 视频状态

    # 获得字幕
    res_text, res_srt, video_state, _, output_dir = mix_recog(video_input, None, hotwords, output_dir)
    # 使用logger输出返回的变量
    logger.info("res_text:{}".format(res_text))
    logger.info("res_srt:{}".format(res_srt))
    logger.info("video_state:{}".format(video_state))
    logger.info("output_dir:{}".format(output_dir))


    # AI裁剪字幕
    prompt_head = ("你是一个视频srt字幕分析剪辑器，输入视频的srt字幕，"
                   "分析其中的精彩且尽可能连续的片段并裁剪出来，输出四条以内的片段，将片段中在时间上连续的多个句子及它们的时间戳合并为一条，"
                   "注意确保文字与时间戳的正确匹配。输出需严格按照如下格式：1. [开始时间-结束时间] 文本，注意其中的连接符是“-”")
    prompt_head2 = ("这是待裁剪的视频srt字幕：")
    llm_result = llm_inference(prompt_head, prompt_head2, res_srt, "gpt-4o-mini-2024-07-18")
    logger.info("llm_result:{}".format(llm_result))

    # 生成视频
    output_dir = '../asset/切片目录/'
    output_dir = os.path.abspath(output_dir)  # 移除路径中的空格
    AI_clip(LLM_res=llm_result, dest_text=res_text, video_spk_input=None, start_ost=0, end_ost=0, video_state=video_state, output_dir=output_dir)
if __name__ == '__main__':
        main()
