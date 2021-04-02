import sys
import pandas as pd
import numpy as np
import sqlalchemy
import pymysql
import pymssql
from datetime import datetime, timedelta
import time

# ======================================================================================================================
# ======================================================================================================================
## Maria DB에서 데이터 Read

def select_maria_result(maria_engine, maria_tname, check_dt):

    maria_conn = maria_engine.connect()

    query = """SELECT * FROM """ + maria_tname + """ WHERE dd = '""" + check_dt + """';"""

    result = pd.read_sql_query(query, maria_conn)

    return result

# ======================================================================================================================
# ======================================================================================================================
def set_line(line, result2):

    ## MS SQL 테이블에 들어갈 라인 정보
    ## 추가 라인 적용 시 추가 필요
    ## #2라인 2개 추가
    if line == 'ABH125_1LINE' :
        result2['LINE'] = 'ABH125c_1'
    elif line == 'ABH250_1LINE' :
        result2['LINE'] = 'ABH250c_1'
    elif line == 'ABH125_2LINE' :
        result2['LINE'] = 'ABH125c_2'
    elif line == 'ABH250_2LINE' :
        result2['LINE'] = 'ABH250c_2'

    ## MS SQL 테이블의 필드 순서와 일치시켜주는 작업
    result2 = result2.loc[:, ['LINE', 'DD', 'HHMISS', 'SRAL_ID', 'POLE', 'AMP', 'BRAND', 'TYP', 'CLS',
                              'POS', 'RST', 'DEFT_TYP', 'X1CODI', 'Y1CODI', 'X2CODI', 'Y2CODI']]

    return result2

# ======================================================================================================================
# ======================================================================================================================
## Vision Maria DB 접속 정보 (From)
maria_host = '165.244.138.62'
maria_user = 'lseadmin'
maria_password = 'Tbvjaosvkdnj00'
maria_db = 'mccb'

maria_engine = sqlalchemy.create_engine('mysql+pymysql://' + maria_user + ':' + maria_password + '@' + maria_host
                                       + ':3306/' + maria_db + '?charset=utf8')

## Vision MSSQL DB 접속 정보 (To)
az_host = '52.231.30.154'
az_user = 'lseadmin'
az_password = 'Tbvjaosvkdnj00'
az_db = 'mccb'

azure_engine = sqlalchemy.create_engine('mssql+pymssql://' + az_user + ':' + az_password + '@' + az_host
                                        + ':21433/' + az_db + '?charset=utf8')

## Line 리스트명
## 추가 라인이 적용될 경우 하기 리스트에 라인 명을 추가해줘야함
## 라인명은 반드시 각 라인 별 생성되는 테이블에 포함된 라인명과 일치해야함
## #2라인 2개 추가
line_list = ['ABH125_1LINE', 'ABH250_1LINE', 'ABH125_2LINE', 'ABH250_2LINE']

## D-1 날짜
## D-1 데이터를 배치로 가져오기 때문에 vat1 는 일반적으로 1
## 변수에 따라 D-n 의 데이터를 가져올 수 있음
var1 = sys.argv[1]
day_var = float(var1)
yesterday = datetime.now() - timedelta(days=day_var)
check_dt = yesterday.strftime('%Y%m%d')
check_mon = check_dt[:-2]

# ======================================================================================================================
# ======================================================================================================================
## 각 라인 별로 수행
for line in line_list:

    maria_tname = maria_db + '.ZMRA_SRFC_INSP_L_' + line + '_' + check_mon
    azure_tname1 = 'ZMRA_SRFC_INSP_L_' + line + '_' + check_mon
    azure_tname2 = 'ZMRA_SRFC_INSP_L'

    try:
        result1 = select_maria_result(maria_engine, maria_tname, check_dt)

    except:
        continue

    if not result1.empty:
        ## MS SQL 테이블의 데이터 타입과 일치시켜주는 작업
        result1['DD'] = result1['DD'].map(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
        result1['HHMISS'] = result1['HHMISS'].map(lambda x: ':'.join([x[0:2], x[2:4], x[4:]]))

        result2 = result1.copy()
        result2 = set_line(line, result2)

        azure_conn = azure_engine.connect()

        try:
            ## 월이 변경되어 신규 테이블 생성이 필요한 경우(라인/월별로 테이블이 존재하기 때문에)
            if not azure_engine.dialect.has_table(azure_engine, azure_tname1):
                create_query = """CREATE TABLE [dbo].[""" + azure_tname1 + """](
                                           [ID] [int] IDENTITY(1,1) PRIMARY KEY NOT NULL,
                                           [DD] [date] NOT NULL,
                                           [HHMISS] [time](0) NOT NULL,
                                           [SRAL_ID] [varchar](50) NULL,
                                           [POLE] [varchar](1) NULL,
                                           [AMP] [varchar](3) NULL,
                                           [BRAND] [varchar](8) NULL,
                                           [TYP] [varchar](8) NULL,
                                           [CLS] [varchar](50) NULL,
                                           [POS] [varchar](8) NULL,
                                           [RST] [varchar](8) NULL,
                                           [DEFT_TYP] [varchar](50) NULL,
                                           [X1CODI] [int] NULL,
                                           [Y1CODI] [int] NULL,
                                           [X2CODI] [int] NULL,
                                           [Y2CODI] [int] NULL);"""

                azure_conn.execute(create_query)

            ## 라인/월 별 테이블
            result1.to_sql(con=azure_engine, name=azure_tname1, index=False, if_exists='append')
            ## 통합 테이블
            result2.to_sql(con=azure_engine, name=azure_tname2, index=False, if_exists='append')

        except Exception as e :
            print(e)
            continue

        else:
            print(line + ' ' + check_dt + ' Insert Compete')
