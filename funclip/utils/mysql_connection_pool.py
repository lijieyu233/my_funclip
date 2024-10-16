import sys

import pymysql
from dbutils.pooled_db import PooledDB

host = '117.72.36.222'
port = 3306
user = 'reed'
password = '2418141009'
database = 'AI_clip'


class MySQLConnectionPool:

    def __init__(self, ):
        self.pool = PooledDB(
            creator=pymysql,  # 使用链接数据库的模块
            mincached=1,  # 初始化时，链接池中至少创建的链接，0表示不创建
            maxconnections=2,  # 连接池允许的最大连接数，0和None表示不限制连接数
            blocking=True,  # 连接池中如果没有可用连接后，是否阻塞等待。True，等待；False，不等待然后报错
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )

    # 获取数据库连接
    def get_conn(self):
        self.conn = self.pool.connection()
        return self.conn

    # 打开数据库连接 cursor 游标 cursor=pymysql.cursors.DictCursor 表示游标返回的数据为字典类型
    def open(self):
        self.conn = self.pool.connection()
        self.cursor = self.conn.cursor(cursor=pymysql.cursors.DictCursor)  # 表示读取的数据为字典类型
        return self.conn, self.cursor

    # 关闭数据库连接
    def close(self, cursor, conn):
        cursor.close()
        conn.close()

    # 查询数据库
    def select_one(self, sql, *args):
        """查询单条数据"""
        conn, cursor = self.open()
        cursor.execute(sql, args)
        result = cursor.fetchone()
        self.close(conn, cursor)
        return result

    def select_all(self, sql, args):
        """查询多条数据"""
        conn, cursor = self.open()
        cursor.execute(sql, args)
        result = cursor.fetchall()
        self.close(conn, cursor)
        return result

    def insert_one(self, sql, args):
        """插入单条数据"""
        self.execute(sql, args, isNeed=True)

    def insert_all(self, sql, datas):
        """插入多条批量插入"""
        conn, cursor = self.open()
        try:
            cursor.executemany(sql, datas)
            conn.commit()
            return {'result': True, 'id': int(cursor.lastrowid)}
        except Exception as err:
            conn.rollback()
            return {'result': False, 'err': err}

    def update_one(self, sql, args):
        """更新数据"""
        self.execute(sql, args, isNeed=True)

    def delete_one(self, sql, *args):
        """删除数据"""
        self.execute(sql, args, isNeed=True)

    def execute(self, sql, args, isNeed=False):
        """
        执行
        :param isNeed 是否需要回滚
        """
        conn, cursor = self.open()
        if isNeed:
            try:
                cursor.execute(sql, args)
                conn.commit()
            except:
                conn.rollback()
        else:
            cursor.execute(sql, args)
            conn.commit()
        self.close(conn, cursor)


def test():
    mysql = MySQLConnectionPool()  # 实例化一个数据库连接池

    sql_insert_one = "insert into `names` (`name`, sex, age) values (%s,%s,%s)"
    mysql.insert_one(sql_insert_one, ('唐三', '男', 25))

    datas = [
        ('戴沐白', '男', 26),
        ('奥斯卡', '男', 26),
        ('唐三', '男', 25),
        ('小舞', '女', 100000),
        ('马红俊', '男', 23),
        ('宁荣荣', '女', 22),
        ('朱竹清', '女', 21),
    ]
    sql_insert_all = "insert into `names` (`name`, sex, age) values (%s,%s,%s)"
    mysql.insert_all(sql_insert_all, datas)

    sql_update_one = "update `names` set age=%s where `name`=%s"
    mysql.update_one(sql_update_one, (28, '唐三'))

    sql_delete_one = 'delete from `names` where `name`=%s '
    mysql.delete_one(sql_delete_one, ('唐三',))

    sql_select_one = 'select * from `names` where `name`=%s'
    results = mysql.select_one(sql_select_one, ('唐三',))
    print(results)

    sql_select_all = 'select * from `names` where `name`=%s'
    results = mysql.select_all(sql_select_all, ('唐三',))
    print(results)


def main():
    mysql_pool = MySQLConnectionPool()
    conn = mysql_pool.get_conn()


if __name__ == '__main__':
    main()
