# CNN-BiLSTM model for Korean NER
CNN과 BiLSTM을 이용한 한국어 개체명 인식기입니다.

#### 사용한 자질은 다음과 같습니다.
- 형태소 non-static word2vec, static word2vec (mecab 사용, gensim으로 word2vec)
- 음절단위 (character cnn)
- POS (mecab 사용)
- 사전정보 (gazette)

#### Requirements
- ```pytorch```
- ```konlpy, mecab```
- ```gensim```

#### 데이터셋
-  엑소브레인 언어분석 말뭉치(ETRI)

#### 성능
![classification_report](NER_practice_Yoo/assets/NER결과Classificaiton_report.png)

#### 결과 예제
![NER_result](NER_practice_Yoo/assets/NER결과.png)

#### 모델
![NER_model](NER_practice_Yoo/assets/NER모델그림.png)  

#### Future work
- CRF + Viterbi
