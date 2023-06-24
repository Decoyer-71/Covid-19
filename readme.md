# Covid-19

## 주제 
    TL(Transfer Learning) 아키텍쳐를 활용한 Covid-19 이미지 분류모델 생성

## 사용언어
<a href="https://www.python.org/" target="_blank"><img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/></a>
<a href="https://jupyter.org/" target="_blank"><img src="https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white"/></a>

## 폴더 분류
[code](https://github.com/Decoyer-71/BrainTumor/tree/master/code) : 학습 및 모델생성 코드


## 1. Data Set
    1) 출처 : [Kaggle Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
    2) 구성
        - 각 Label(COVID, Lung_Opacity, Normal, Viral Pneumonia) 폴더 별 images와 masks 폴더로 구성
        - images 폴더의 데이터만 활용

## 2. 프로젝트 개요
    1) 목표 : Testing 결과 acc 0.90 이상 정확도 
    2) 소스명 : 
    3) 개발환경 
        - data폴더 -> env_list.txt
        - Google colab(fitting 용)
        
## 아키텍처별 실험(Goole colab)
    - 공통사항 : FC계층에 대해 세부 수치변경을 통해 목표달성 모델 생성(FineTuning)
### 1) Mobilenet 
        (1) Hiddenlayer, Dropout 미설정 / Optimizer : Adam(1e-4) / epochs : 30
            가. Evaluate 결과 : loss: 0.7634, acc: 0.9360
            나. 소요시간 : 0:35:41
            다. 평가 : 과적합 심함, validation_acc 그래프로 보아 learning_rate 조절이 필요함
![image](https://github.com/Decoyer-71/Covid-19/assets/127948197/64b15dfd-7675-40b6-8c8a-298782f0ee2f)

        (2) Hiddenlayer : 3(node : 128), Dropout : 0.25 / Optimizer : Adam(1e-5) / epochs : 30
            가. Evaluate 결과 : loss 0.5023, acc 0.9246
            나. 소요시간 : 0:34:58
            다. 평가 : validation_acc 그래프가 안정적이고 정확도가 90%를 넘으나, 여전히 과적합이 심하다.
![image](https://github.com/Decoyer-71/Covid-19/assets/127948197/deda14fb-f5d4-4ac0-afba-cab8aa901bae)

        (3) Hiddenlayer : 3(node : 128), Dropout : 0.5 / Optimizer : Adam(1e-5) / epochs : 35
            가. Evaluate 결과 : loss 0.5867, acc 0.9279
            나. 소요시간 : 0:39:0
            다. 평가 : 정확도 92%, 과적합을 이전에 비해서 약간 해소하였음.
![image](https://github.com/Decoyer-71/Covid-19/assets/127948197/7d240a90-cce1-4864-b996-aba98b27efa0)

### 2) Xception
        (1) Hiddenlayer, Dropout 미설정 / Optimizer : Adam(1e-4) / epochs : 30
            가. Evaluate 결과 : loss 0.5003,  acc 0.9232
            나. 소요시간 : 1:59:25
            다. 평가 : 과적합 심함, 목표 점수는 넘었으나 mobilnet과 마찬가지로 node를 추가하면서 dropout으로 과적합을 줄일 필요가 있음.
![image](https://github.com/Decoyer-71/Covid-19/assets/127948197/8f0c38d0-ea9d-466c-bab1-acf5bb4bd070)

        (2) Hiddenlayer : 3(node : 128), Dropout : 0.25 / Optimizer : Adam(1e-4) / epochs : 20
            가. Evaluate 결과 : loss 0.6268, acc 0.9293
            나. 소요시간 : 1:21:23
            다. 평가 :  시간 소요를 줄이기 위해 epoch는 20으로 조정, 목표점수를 달성했으나 과적합이 심함
![image](https://github.com/Decoyer-71/Covid-19/assets/127948197/10ae9e7d-6349-48b1-b658-bb4354004bac)

        (3) Hiddenlayer : 3(node : 128), Dropout : 0.5 / Optimizer : Adam(1e-4) / epochs : 20
            가. Evaluate 결과 : loss 0.7406, acc 0.9230
            나. 소요시간 : 1:21:22
            다. 평가 :  과적합이 조금 줄었으나, 크게 해소되지 않음. learning_rate를 더 높여서 시도해볼 예정
![image](https://github.com/Decoyer-71/Covid-19/assets/127948197/e61879d8-1705-4522-9bf9-f536c04021ec)
            


## 결론
