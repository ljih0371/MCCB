import os
import pandas as pd
import numpy as np
import sys
import sqlalchemy
import pymssql
import re
from azure.storage.blob import BlockBlobService
from datetime import datetime, timedelta

# ======================================================================================================================
# ======================================================================================================================
## Blob Storage 정보
STORAGEACCOUNTNAME = 'cj01vision001'
STORAGEACCOUNTKEY = 'T7Q7tncoj0Er29cvCK6DsU9w0gnM0tvfYoWOc1Btacx+TySBxPdx+h+n3YUghgpRpZUcSfkHiGgLX98wGDXnmA=='
CONTAINERNAME = 'mccb'

## Vision MSSQL DB 접속 정보
az_host = '52.231.30.154'
az_user = 'lseadmin'
az_password = 'Tbvjaosvkdnj00'
az_db = 'mccb'
azure_engine = sqlalchemy.create_engine('mssql+pymssql://' + az_user + ':' + az_password + '@' + az_host
                                        + ':21433/' + az_db + '?charset=utf8')

## 라인 리스트
## 추가 라인이 적용될 경우 하기 리스트에 라인 명을 추가해줘야함
## 라인명은 반드시 Blob의 라인 폴더 명과 일치해야함
## #2라인 2개 추가
line_list = ['ABH125c_1', 'ABH250c_1', 'ABH125c_2', 'ABH250c_2']
#line_list = ['ABH125c_1']

## 사이드 리스트
## 라인명은 반드시 Blob의 사이드 폴더 명과 일치해야함
side_list = ['RIGHT', 'LOAD', 'LOAD_TAP', 'TOP', 'LINE_TAP', 'LINE', 'LEFT' ]

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
## NG_XML 폴더에 있는 불량 바운딩 이미지(.jpg) 파일 리스트를 추출

def allblobs(line, side_list, check_dt, check_mon):
    fpath_list = []

    for side in side_list:

        PREFIX = line + '/NG_XML' + '/' + check_mon + '/' + side + '/' + check_dt

        try :
            blob_list = block_blob_service.list_blobs(CONTAINERNAME, prefix=PREFIX)
        except :
            continue
        else :
            for blob in blob_list:
                fpath_list.append(blob.name)

    return fpath_list

# ======================================================================================================================
# ======================================================================================================================
## NG_XML 파일이 저장된 Blob Storage 경로 정보를 생성하고, DB에 Insert 하기 위한 구조로 변경

def set_structure(line, fpath_list):
    result_temp = pd.DataFrame()

    for fpath in fpath_list:
        fullpath = 'https://cj01vision001.blob.core.windows.net/mccb/' + fpath
        fname = fpath.split('/')[4]

        fdate = fname.split('.')[0].split('_')[0]
        ftime = fname.split('.')[0].split('_')[1]
        bc = fname.split('.')[1]
        fside = fname.split('.')[2]
        deft = fname.split('.')[3]
        x1 = fname.split('.')[4].split('_')[0]
        y1 = fname.split('.')[4].split('_')[1]
        x2 = fname.split('.')[4].split('_')[2]
        y2 = fname.split('.')[4].split('_')[3]

        diction = {'LINE': line,
                   'DD': fdate,
                   'HHMISS': ftime,
                   'SRAL_ID': bc,
                   'POS': fside,
                   'DEFT_TYP': deft,
                   'X1CODI': x1,
                   'Y1CODI': y1,
                   'X2CODI': x2,
                   'Y2CODI': y2,
                   'URL': fullpath
                   }

        result_temp = result_temp.append(diction, ignore_index=True)

    return result_temp

# ======================================================================================================================
# ======================================================================================================================
## Connect to Blob
block_blob_service = BlockBlobService(STORAGEACCOUNTNAME, STORAGEACCOUNTKEY)

result = pd.DataFrame()

## 각 라인 별로 수행
for line in line_list:
    print('>> '+ line + ' : 시작 '+ check_dt)
    try:
        fpath_list = allblobs(line, side_list, check_dt, check_mon)
        if fpath_list:
            result = set_structure(line, fpath_list)

            ## MS SQL 테이블의 필드 순서와 일치시켜주는 작업
            result = result.loc[:, ['LINE', 'DD', 'HHMISS', 'SRAL_ID', 'POS', 'DEFT_TYP', 'X1CODI', 'Y1CODI', 'X2CODI', 'Y2CODI', 'URL']]

            ## MS SQL 테이블의 데이터 타입과 일치시켜주는 작업
            result['DD'] = result['DD'].map(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
            result['HHMISS'] = result['HHMISS'].map(lambda x: ':'.join([x[0:2], x[2:4], x[4:]]))
            result['X1CODI'] = result['X1CODI'].astype(int)
            result['Y1CODI'] = result['Y1CODI'].astype(int)
            result['X2CODI'] = result['X2CODI'].astype(int)
            result['Y2CODI'] = result['Y2CODI'].astype(int)

            ## MS SQL 테이블에 Append
            result.to_sql(con=azure_engine, name='NG_IMG_BPATH', index=False, if_exists='append')
            print(line + ' MS SQL 테이블에 Append 완료')

    except Exception as e:
        print(line+' exception : '+str(e))
        continue



