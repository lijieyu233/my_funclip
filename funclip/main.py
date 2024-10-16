import json
import os
import sys

import pymysql.cursors
from moviepy.video.io.VideoFileClip import VideoFileClip

from funclip.llm.openai_api import my_openai_call
from funclip.utils.logger_setup import setup_logger
from funclip.utils.mysql_connection_pool import MySQLConnectionPool
from funclip.utils.trans_utils import extract_timestamps
from videoclipper import VideoClipper
import argparse
from funasr import AutoModel

logger = setup_logger()
mysql_pool = MySQLConnectionPool()
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
def video_recog(video_input, sd_switch, hotwords):  # 从视频中提取音频处理
    return audio_clipper.video_recog(video_input, sd_switch, hotwords)


# # 提取音频 或从视频中提取音频生成srt
# def mix_recog(video_input, audio_input, hotwords, output_dir='../asset/切片目录'):
#     # 输出路径处理
#     output_dir = output_dir.strip()  # 清理输出路径中的多余空白
#     logger.info('mix_recog:用户指定了输出路径,输出路径为:' + output_dir)
#     output_dir = os.path.abspath(output_dir)  # 将相对路径转换为绝对路径
#     audio_state, video_state = None, None  # 初始化状态
#
#     # 优先调用视频
#     if video_input is not None:
#         res_text, res_srt, video_state = video_recog(
#             video_input, 'No', hotwords, output_dir=output_dir)  # 调用视频识别
#         # todo 此地生成的输出路径output_dir 保存到全局变量
#         return res_text, res_srt, video_state, None, output_dir  # 返回视频识别结果和状态


# todo 2. AI推理 划分字幕
def llm_inference(system_content, user_content, srt_text, model,
                  apikey="sk-MIBD51NOevuLr38Pj9XRmS7xMrm8oprTIyv312FNscHSjcpo"):
    SUPPORT_LLM_PREFIX = ['qwen', 'gpt', 'g4f', 'moonshot']
    if model.startswith('gpt') or model.startswith('moonshot'):
        return my_openai_call(apikey, model, system_content, user_content + '\n' + srt_text)
    else:
        logger.error("LLM name error, only {} are supported as LLM name prefix."
                     .format(SUPPORT_LLM_PREFIX))


# todo 3.根据AI生成后的字幕 划分视频 进行剪辑
def AI_clip(LLM_res,video,video_name, sentences,dest_text, video_spk_input, start_ost, end_ost, output_dir):  #
    timestamp_list = extract_timestamps(LLM_res)  # 提取生成后字幕的时间戳
    logger.info("timestamp_list:{}".format(timestamp_list))
    output_dir = output_dir.strip()  # 移除路径中的空格
    if not len(output_dir):  # 如果路径为空，将 output_dir 设置为 None
        output_dir = None  # 如果路径为空，将 output_dir 设置为 None
    else:
        output_dir = os.path.abspath(output_dir)  # 转换为绝对路径

    audio_clipper.my_video_clip(  # 调用自定义的视频剪辑函数
        video_name=video_name,  # 原始字幕
        start_ost=start_ost,
        end_ost=end_ost,
        video=video,
        sentences=sentences,
        output_dir=output_dir,  # 输出目录
        timestamp_list=timestamp_list)  # AI字幕时间戳


def fetch_data_from_db(criterion):
    if criterion == 'generate_srt_and_sentences':
        # 查询数据库
        with mysql_pool.get_conn() as conn:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = '''select id,video_local_path from video limit 1'''
                cursor.execute(sql)
                videos = cursor.fetchall()
                return videos
    elif criterion == 'AI_slice':
        with mysql_pool.get_conn() as conn:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = '''select id,video_local_path,sentences,srt_local_path from video where is_transcribed = 1 limit 1'''
                cursor.execute(sql)
                videos = cursor.fetchall()
                return videos

def generate_srt_and_sentences():
    videos = fetch_data_from_db('generate_srt_and_sentences')  # 从数据库中获取视频数据
    id = videos['id']
    video_local_path = os.path.abspath(videos['video_local_path'])  # 获取视频文件的绝对路径
    logger.info(f'id:{id}')
    logger.info(f'video_local_path:{video_local_path}')
    video = VideoFileClip(video_local_path)  # 创建VideoFileClip对象

    # 获取文件名称，创建保存路径
    video_name = os.path.splitext(video_local_path)[0]
    video_name = os.path.basename(video_name)
    logger.info(f'video_name:{video_name}')

    srt_output_dir = 'D:/ljy_folder/my_funclip/output/srt'
    os.makedirs(srt_output_dir, exist_ok=True)

    # 获得字幕
    res_text, res_srt, video_state = video_recog(video_input=video_local_path, hotwords='',
                                                 sd_switch='No')

    # 获取srt sentences
    logger.info(f'res_srt:{res_srt}')
    logger.info(f'video_state:{video_state}')
    sentences = video_state['sentences']
    logger.info(f'json前sentences:{sentences}')
    sentences = json.dumps(sentences, ensure_ascii=False)
    logger.info(f'sentences:{sentences}')

    # 将字幕存入文件，地址保存到数据库
    srt_save_path = os.path.join(os.path.abspath(srt_output_dir), f'{video_name}.srt')
    logger.info(f'srt_save_path:{srt_save_path}')
    with open(srt_save_path, 'w', encoding='gbk') as f:
        f.write(res_srt)

    # 存入数据库
    with mysql_pool.get_conn() as conn:
        with conn.cursor() as cursor:
            sql = '''update video 
                set srt_local_path=%s,
                sentences=%s,
                is_transcribed=True
                where id=%s'''
            logger.info(f'Executing SQL with values:srt_save_path={srt_save_path},sentencessql:{sql}')
            cursor.execute(sql, (srt_save_path, sentences, id))
            conn.commit()


def AI_slice():
    video_infos = fetch_data_from_db('AI_slice')  # 从数据库中获取视频数据
    video_info=video_infos[0]
    id = video_info['id']
    video_local_path = os.path.abspath(video_info['video_local_path'])
    sentences = video_info['sentences']
    srt_local_path = video_info['srt_local_path']
    logger.info(f'id:{id}')
    logger.info(f'video_local_path:{video_local_path}')
    logger.info(f'sentences:{sentences}')
    logger.info(f'srt_local_path:{srt_local_path}')
    video = VideoFileClip(video_local_path)  # 创建VideoFileClip对象


    #获取sentences
    sentences = json.loads(sentences)

    #从srt_local_path中获取srt 赋值到变量video_srt中
    with open(srt_local_path,'r',encoding='gbk') as f:
        video_srt = f.read()
        # AI裁剪字幕


    # llm推理
    prompt_head = ("你是一个视频srt字幕分析剪辑器，输入视频的srt字幕，"
                   "分析其中的精彩且尽可能连续的片段并裁剪出来，输出四条以内的片段，将片段中在时间上连续的多个句子及它们的时间戳合并为一条，"
                   "注意确保文字与时间戳的正确匹配。输出需严格按照如下格式：1. [开始时间-结束时间] 文本，注意其中的连接符是“-”")
    prompt_head2 = ("这是待裁剪的视频srt字幕：")
    llm_result = llm_inference(prompt_head, prompt_head2, video_srt, "gpt-4o-mini-2024-07-18")
    logger.info("llm_result:{}".format(llm_result))

    #
    video_name = os.path.splitext(video_local_path)[0]
    video_name = os.path.basename(video_name)

    video_output_dir = 'D:/ljy_folder/my_funclip/output/video'
    video_output_dir = os.path.abspath(video_output_dir)

    video_clip_save_dir = os.path.join(video_output_dir, f'{video_name}')
    os.makedirs(video_clip_save_dir, exist_ok=True)
    logger.info(f'video_clip_save_dir:{video_clip_save_dir}')

    AI_clip(video=video,video_name=video_name,LLM_res=llm_result, dest_text=video_srt, video_spk_input=None, start_ost=0, end_ost=0,
             output_dir=video_clip_save_dir,sentences=sentences)

def main():
    videos = fetch_data_from_db()  # 从数据库中获取视频数据
    id = videos['id']
    video_input = '../asset/测试视频/test.mp4'  # 输入视频路径
    video = VideoFileClip(video_input)

    # 获取文件名称，创建保存路径
    video_name = os.path.splitext(video_input)[0]
    video_name = os.path.basename(video_name)
    logger.info(f'video_name:{video_name}')

    video_output_dir = 'D:/ljy_folder/my_funclip/output/video'
    srt_output_dir = 'D:/ljy_folder/my_funclip/output/srt'
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(srt_output_dir, exist_ok=True)

    hotwords = ''  # 热词
    output_dir = '../asset/测试视频/output'  # 输出路径

    res_text = ''  # 识别文本结果
    res_srt = ''  # 识别srt结果
    video_state = None  # 视频状态

    # 获得字幕
    # res_text, res_srt, video_state, _, output_dir = mix_recog(video_input, None, hotwords, output_dir)
    res_text, res_srt, video_state = video_recog(video_input=video_input, hotwords='', output_dir=output_dir,
                                                 sd_switch='No')
    logger.info(f'res_srt:{res_srt}')
    logger.info(f'video_state:{video_state}')
    sentences = video_state['sentences']
    logger.info(f'json前sentences:{sentences}')
    sentences = json.dumps(sentences, ensure_ascii=False)
    logger.info(f'sentences:{sentences}')

    # 将字幕存入文件，地址保存到数据库
    srt_save_path = os.path.join(os.path.abspath(srt_output_dir), f'{video_name}.srt')
    logger.info(f'srt_save_path:{srt_save_path}')
    with open(srt_save_path, 'w', encoding='gbk') as f:
        f.write(res_srt)

    # 存入数据库
    with mysql_pool.get_conn() as conn:
        with conn.cursor() as cursor:
            sql = '''update video 
            set srt_local_path=%s,
            sentences=%s,
            is_transcribed=True
            where id=%s'''
            logger.info(f'Executing SQL with values:srt_save_path={srt_save_path},sentencessql:{sql}')
            cursor.execute(sql, (srt_save_path, sentences, id))
            conn.commit()

    # 使用logger输出返回的变量
    # logger.info("res_text:{}".format(res_text))
    # logger.info("res_srt:{}".format(res_srt))
    # logger.info("video_state:{}".format(video_state))
    # logger.info("output_dir:{}".format(output_dir))
    # logger.info(f'获得字幕后的video_state:{video_state}')

    # AI裁剪字幕
    prompt_head = ("你是一个视频srt字幕分析剪辑器，输入视频的srt字幕，"
                   "分析其中的精彩且尽可能连续的片段并裁剪出来，输出四条以内的片段，将片段中在时间上连续的多个句子及它们的时间戳合并为一条，"
                   "注意确保文字与时间戳的正确匹配。输出需严格按照如下格式：1. [开始时间-结束时间] 文本，注意其中的连接符是“-”")
    prompt_head2 = ("这是待裁剪的视频srt字幕：")
    llm_result = llm_inference(prompt_head, prompt_head2, res_srt, "gpt-4o-mini-2024-07-18")
    logger.info("llm_result:{}".format(llm_result))

    #
    video_output_dir = os.path.abspath(video_output_dir)
    video_clip_save_dir = os.path.join(video_output_dir, f'{video_name}')
    os.makedirs(video_clip_save_dir, exist_ok=True)
    logger.info(f'video_clip_save_dir:{video_clip_save_dir}')
    # 生成视频
    AI_clip(LLM_res=llm_result, dest_text=res_text, video_spk_input=None, start_ost=0, end_ost=0,
            video_state=video_state, output_dir=video_clip_save_dir)


if __name__ == '__main__':
    AI_slice()
    # generate_srt_and_sentences()
    # main()
