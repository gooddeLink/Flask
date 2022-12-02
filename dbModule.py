import pymysql
from dotenv import load_dotenv
import os 

# load .env
load_dotenv()

class Database():
    def __init__(self):
        self.db = pymysql.connect(host='database-2.cc5um8ekupdb.ap-northeast-2.rds.amazonaws.com',
                                  user='admin',
                                  password=os.environ.get('DATABASE_PW'),
                                  db='LC',
                                  charset='utf8')
        self.cursor = self.db.cursor(pymysql.cursors.DictCursor)
 
    def execute(self, query, args={}):
        self.cursor.execute(query, args)  
 
    def executeOne(self, query, args={}):
        self.cursor.execute(query, args)
        row = self.cursor.fetchone()
        return row
 
    def executeAll(self, query, args={}):
        self.cursor.execute(query, args)
        row = self.cursor.fetchall()
        return row
 
    def commit(self):
        self.db.commit()
