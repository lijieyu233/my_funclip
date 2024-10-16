CREATE DATABASE my_database CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
create database AI_clip  character set utf8mb4 collate utf8mb4_general_ci;
use AI_clip;

create table if not exists `video`
(
    `id`          int auto_increment primary key Comment '主键id',
    `name`        varchar(256) not null Comment '视频名称',
    is_transcribed tinyint(1) default 0 Comment '是否已生成字幕',
    is_edited      tinyint(1) default 0 Comment '是否已经切片',
    srt_local_path varchar(256) Comment '字幕本地路径',
    video_state json Comment '视频状态'
);

update video set is_transcribed = 1 and sentences=

select * from video where TRUE;
delete from video where id = 1;
insert into `video` (video_local_path) values('D:/ljy_folder/my_funclip/raw_video/test.mp4')