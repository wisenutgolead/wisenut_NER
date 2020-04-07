# CNN-BiLSTM + CRF model for Korean NER
CNN-BiLSTM+CRF를 이용한 한국어 개체명 인식기 프로젝트


## 환경 구성

OS: Windows 10  
IDE: Pycharm  
venv: Anaconda  
Language: Python 3.6  

### Pycharm + Anaconda
[PyCharm for Anaconda Community Edition](https://www.jetbrains.com/pycharm/download/download-thanks.html?code=PCC&platform=windowsAnaconda)

**miniconda 설치**

**Pycharm + Anaconda 연동**  
Pycharm - Files- Settings - Project interpreter에서 Anaconda 가상환경 생성 및 추가

![first step](img/Pycharm_1.png)
![second step](img/Pycharm_2.png)
![third step](img/Pycharm_3.png)

### Requirements

Anaconda Prompt 사용  

Pytorch : 딥러닝 라이브러리  
 ```
 conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
 ```

gensim : 임베딩 라이브러리  
```
pip install gensim
```

eunjeon : 형태소 분석기 Windows용 Mecab  
1. `pip install JPype1`  
2. `pip install eunjeon`

Sklearn : 머신러닝 라이브러리  
`pip install scikit-learn`

joblib : 학습 모델 저장 라이브러리  
`pip install joblib`

## Model Design
![NER_model](img/scheme.png)

