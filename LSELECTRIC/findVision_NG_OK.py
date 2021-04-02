
import os, glob
import shutil
import sys
from datetime import datetime, timedelta


# ======================================================================================================================
# ======================================================================================================================
## Samba 경로
origin_path = '/data/mccb/'

## 라인 리스트
## 추가 라인이 적용될 경우 하기 리스트에 라인 명을 추가해줘야함
## 라인명은 반드시 Samba의 라인 폴더 명과 일치해야함
## 신규 확장 라인 #2 2개 추가
line_list = ['ABH125c_1', 'ABH250c_1', 'ABH125c_2', 'ABH250c_2']

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
## NG폴더를 검색하여 NG 파일 리스트를 Read하고, 파일명에서 Barcode 명을 추출하여 리스트로 만듦
def all_ng_barcode(ng_path, check_dt):
    ng_bc_list = []

    for root, dirs, files in os.walk(ng_path):
        if files:
            condition = check_dt + '*.*.*.*.png'
            condition_path = os.path.join(root, condition)
            dt_files = glob.glob(condition_path)
            for file in dt_files:
                ng_bc_list.append(os.path.basename(file).split('.')[1])

    return ng_bc_list

# ======================================================================================================================
# ======================================================================================================================
## NG Barcode 리스트로 OK폴더를 검색하여 해당 바코드의 OK 파일을 찾아서 NG_OK 폴더로 복사
def files_to_copy(ok_path, bc_list, check_dt, ngok_path):
    for root, dirs, files in os.walk(ok_path):
        rootpath = os.path.abspath(root)
        for file in files:
            if file.startswith(check_dt) and file.endswith('png') and any(bc in file for bc in bc_list):
                filepath = os.path.join(rootpath, file)
                side_dir = os.path.basename(os.path.normpath(rootpath))

                to_path = ngok_path + '/' + side_dir
                if not os.path.exists(to_path):
                    os.makedirs(to_path)

                try:
                    shutil.copy(filepath, to_path + '/' + file)
                except:
                    continue
            else:
                continue

# ======================================================================================================================
# ======================================================================================================================
## 각 라인 별로 수행
for line in line_list:
    print('>> '+ line + ' : 시작 ' + check_dt)
    ng_path = origin_path + '/' + line + '/NG/' + check_mon
    ok_path = origin_path + '/' + line + '/OK/' + check_mon
    ngok_path = origin_path + '/' + line + '/NG_OK/' + check_mon

    try:
        ## NG 바코드 리스트 추춣
        ng_bc_list = all_ng_barcode(ng_path, check_dt)
        print(line + '  NG 바코드 수 : ' + str(len(ng_bc_list)))

        ## NG 바코드를 가지고 OK 파일 검색 및 복사
        files_to_copy(ok_path, ng_bc_list, check_dt, ngok_path)
        print(line + ' OK 파일 검색 및 복사 완료')

    except Exception as e:
        print(line+' exception : '+e)
        continue
