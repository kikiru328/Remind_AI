# 프로그램 실행 전 준비사항

## 1. Anaconda 프로그램 설치, 엑셀 프로그램 필요.
# 기타) conda 가상환경 생성 및 접속, python=3.8

# boneage 폴더로 들어가기. ex) >cd C:\boneage

## 2. Anaconda prompt 에서 추가 패치기 설치 
# >pip install --user -r requirements.txt 이용. 5분정도 소요됨.

## 3. 프로그램 실행방법
# > C:\boneage  (폴더 안에서)
# >python bone_age.py    20초 정도 소요됨. (GUI 프로그램)



*** 이미지는 폴더 내에 있는 sample.jpg를 활용하여 test 해볼 수 있습니다. ***
     sample.jpg의 info는 아래와 같습니다.
	1. 이름: 홍길동(변경가능)
	2. 나이: 6 (변경 될경우 신장예측이 비정상 값(abnormal)이 도출될 수 있음)
	3. 신장: 115 (변경 될경우 신장예측이 비정상 값(abnormal)이 도출될 수 있음)
	4. 성별: Male (변경 될경우 신장예측이 비정상 값(abnormal)이 도출될 수 있음)