## EURON 2기 - 스터디팀 19주차
<details>
<summary>CV</summary>
<div markdown="1">       
  
  <br />

  <br />
  
</div>
</details>


<details>
<summary>DA</summary>
<div markdown="1">       

<br />  
  
## 기말고사 휴식기간 입니다. 
  
</div>
</details>


<details>
<summary>NLP</summary>
<div markdown="1">       

| 주차 | 내용             | 발표자                               | 발표자료 |
| ---- | ---------------- | ------------------------------------ | -------- |
| 19    | cs224n 19강     | 임세영, 황채원          | [📚]()    |

## Assignment
  
### 📍 예습과제(~7/11)
  
1️⃣ CS224N 19강을 수강하고, 요약 및 정리한 내용을 깃허브에 업로드

2️⃣ (선택) 질문 사항이나 공유하고 싶은 내용 깃허브 issue에 추가
- 과제 제출 방법
    - 레포: (origin) Ewha-Euron/2022-1-Euron-NLP
    - issue 추가
        - 제목: [19주차] 질문 있습니다/~ 내용 공유합니다.
        - label:
            - 강의 내용 중 이해가 잘 되지 않는 부분 `question`
            - 강의에는 없지만 추가로 궁금한 사항 `question`
            - 강의에는 없지만 추가로 공유하고 싶은 내용 `share`

### 예습과제 제출 방법
  
> 해당 파일을 `master` branch에 업로드하신 후 해당 `master`  branch에서  `pull request` 를 진행해주세요.
  
- 과제 제출 방법
    - 레포: (origin) username/2022-1-Euron-Study-Assignments
    - 브랜치: `master`
    - 해당 주차 브랜치에 과제 업로드하고 Pull Request, 이때 label은 `예습과제`
  
### 📍 복습과제(~7/11)

1️⃣ 아래 구글 드라이브에서 ipynb 파일을 다운받아 필사 과제를 진행해주시면 됩니다.
  
  - [NLG 실습](https://colab.research.google.com/drive/1ohXyJfwK3rCBas2HtmvTtljLjVKdj2Z7?usp=sharing)
  
### 복습과제 제출 방법
  
> 해당 파일을 `Week_19` branch에 업로드하신 후 해당 `Week_19`  branch에서  `pull request` 를 진행해주세요.
  
- 과제 제출 방법
    - 레포: (origin) username/2022-1-Euron-Study-Assignments
    - 브랜치: `Week_19`
    - 해당 주차 브랜치에 과제 업로드하고 Pull Request, 이때 label은 `NLP` , `복습과제`
  

## Due
  
📍 **7월 11일**까지 제출합니다.
  
📍 18강에 해당하는 복습과제는 다음주에 할당 될 예정입니다. 

</div>
</details>





<details>
<summary>CP</summary>
<div markdown="1">       

| 주차 | 내용             | 발표자                               | 발표자료 |
| ---- | ---------------- | ------------------------------------ | -------- |
| 19    | NLP : [자연어 기반 기후기술분류 AI 경진대회](https://dacon.io/competitions/official/235744/overview/description)     | 한예송, 홍재령        | [📚]()    |
 
  
## Assignment
  
### 📍 예습과제(~7/14)

* 국가 연구개발과제를 '기후기술분류체계'에 맞추어 라벨링하는 알고리즘 개발하는 대회로 award 노트북은 공개되어있지 않으나, 아래의 키워드를 중심으로 대회에 공개된 노트북들을 공부하시면 좋을 것 같습니다. 
  
  
  
1️⃣ Baseline, EDA 
  
👉 Baseline code [코드 공유 상단 3개](https://dacon.io/competitions/official/235744/codeshare)
  
① BERT tokenizer + classifier 

② okt Tokenizer + CounterVectorizer + Rnadomforest Classifier

② okt Tokenizer + keras embedding + LSTM
 
  
👉 [EDA](https://dacon.io/competitions/official/235744/codeshare/3008?page=1&dtype=recent)   
  
① [AutoEDA](http://statwith.com/autoeda-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%ED%8C%A8%ED%82%A4%EC%A7%80%EC%9D%98-%EC%9E%90%EB%8F%99%ED%99%94-%ED%83%90%EC%83%89%EC%A0%81-%EC%9E%90%EB%A3%8C%EB%B6%84%EC%84%9D-%EB%8F%84%EA%B5%AC-%EC%86%8C/)

  * 중복 데이터, 레이블 불균형을 빠르게 확인 
  
② Text preprocessing : [KoBERT Tokenizer](https://github.com/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb)
  
  
2️⃣ PORORO library 
  
👉 [PORORO Sentence Embedding 사용 노트북](https://dacon.io/competitions/official/235744/codeshare/3305?page=1&dtype=recent)

* [카카오브레인 PORORO 라이브러리](https://kakaobrain.github.io/pororo/)
  

  
  
3️⃣ XLM-RoBERTa 다국어 사전학습 모델 

👉 [Private 7위 노트북](https://dacon.io/competitions/official/235744/codeshare/3099?page=1&dtype=recent)
  
  * Text Embedding : BERT, word2vec, BoW 등 다양한 임베딩 방법을 다양한 input data 결합에 대해 시도 
  * Classifier : BERT, Logistic, LightGBM 모델 결과 앙상블 
  
👉 [한국어 적용 관련 논문](https://repository.hanyang.ac.kr/handle/20.500.11754/153286) 
 
👉 [데이콘 내 실습코드](https://dacon.io/en/competitions/official/235875/codeshare/4539?page=1&dtype=recent)
  

➕ [Text classification google guide](https://developers.google.cn/machine-learning/guides/text-classification?hl=zh-cn)  
  

### 예습과제 제출 방법
  
> 해당 파일을 `master` branch에 업로드하신 후 해당 `master`  branch에서  `pull request` 를 진행해주세요.
  
- 과제 제출 방법
    - 레포: (origin) username/2022-1-Euron-Study-Assignments
    - 브랜치: `master`
    - 해당 주차 브랜치에 과제 업로드하고 Pull Request, 이때 label은 `예습과제`
  



</div>
</details>
