from moviepy.video.io.VideoFileClip import VideoFileClip
import torch

def mian():
    video=VideoFileClip('../../asset/测试视频/《静夜思》的结构主义解读.mp4')
    new_video=video.subclip(0,60)
    new_video.write_videofile("./test.mp4")

def test():
    print(torch.__version__)
if __name__ == '__main__':
    test()




# 生成一个mysql连接
conn = pymysql.connect(host='localhost', port=3306, user='root', password='123456', database='test')
cur = conn.cursor()
print("数据库版本:", cur.execute("select version();"))
logging.info("数据库版本:", cur.execute("select version();"))
print("数据库版本:", cur.fetchone())

# 生成一个Python函数，用于二分查找数组中的某个元素
