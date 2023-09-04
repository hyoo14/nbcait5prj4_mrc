# Open-Domain Question Answering

MRC (Machine Reading Comprehension) is a task that involves understanding a given passage and inferring the answer to a provided question.

#Boostcamp 5th #NLP

Period| 2023.06.07 ~ 2023.06.22 19:00

## Overview

"서울의 GDP는 세계 몇 위야?", "MRC가 뭐야?"

우리는 궁금한 것들이 생겼을 때, 아주 당연하게 검색엔진을 활용하여 검색을 합니다. 이런 검색엔진은 최근 기계독해 (MRC, Machine Reading Comprehension) 기술을 활용하며 매일 발전하고 있는데요. 본 대회에서는 우리가 당연하게 활용하던 검색엔진, 그것과 유사한 형태의 시스템을 만들어 볼 것입니다.

Question Answering (QA)은 다양한 종류의 질문에 대해 대답하는 인공지능을 만드는 연구 분야입니다.  
다양한 QA 시스템 중, Open-Domain Question Answering (ODQA) 은 주어지는 지문이 따로 존재하지 않고 사전에 구축되어있는 Knowledge resource 에서 질문에 대답할 수 있는 문서를 찾는 과정이 추가되기 때문에 더 어려운 문제입니다.

![](https://lh3.googleusercontent.com/Y3xnzbSr2ToromE00q9qok5VcB2KVPKfpufvbQqdKjZ-OGEX01Wk1TSPaOAcz2r4J301dfNYBmoto1_tFvi2Fjegy4xHLAH-KCSBtxjAOkJDyzK6sTVgRQXeD_-iPEZNdQVPUvqNu4JkoKUe2QYHYWg)

본 ODQA 대회에서 우리가 만들 모델은 two-stage로 구성되어 있습니다. 첫 단계는 질문에 관련된 문서를 찾아주는 "retriever" 단계이고, 다음으로는 관련된 문서를 읽고 적절한 답변을 찾거나 만들어주는 "reader" 단계입니다. 두 가지 단계를 각각 구성하고 그것들을 적절히 통합하게 되면, 어려운 질문을 던져도 답변을 해주는 ODQA 시스템을 여러분들 손으로 직접 만들어보게 됩니다.

따라서, 대회는 더 정확한 답변을 내주는 모델을 만드는 팀이 좋은 성적을 거두게 됩니다.

![](https://lh6.googleusercontent.com/KWJC9GhH6glatYmU9-ufMR5PGFguEP5zKBH5tmcFIw1YjdN0mQjOlwdSDTV69e54HmGmSomDxDpyduX4q-t_a3mG1HP0ZxsrPnxMuKDEEgiqyhvAcAtIYqpTWdompLnM7w7Lz-SrLyTpwg56kvfcPIo)

최종적으로 테스트해야하는 결과물은

- input: Query만 주어집니다.

- output: 주어진 Query에 알맞는 string 형태의 답안

### Evaluation Metric

두 가지 평가지표를 사용합니다.

1. Exact Match (EM): 모델의 예측과, 실제 답이 정확하게 일치할 때만 점수가 주어집니다. 즉 모든 질문은 0점 아니면 1점으로 처리됩니다. 단, 띄어쓰기나 "."과 같은 문자가 포함되어 있다고 오답으로 처리되면 억울하겠죠? 이런 것은 제외한 후 정답에 대해서만 일치하는지 확인합니다. 또한 답이 하나가 아닐 수 있는데, 이런 경우는 하나라도 일치하면 정답으로 간주합니다.

![](https://lh6.googleusercontent.com/WOfzeHPqJ-2hdSeW6_ICswnzI4obOFhMQxumly15C4sdZ7gB2M1k9RQnZ8uAVmMcpokOHnA9efyqBjB5dDGZaxsUSsXpzx-cmlaUSNb6gBIK7IWTAxXaNApt6vtTDD40g8XRECgEUsZX8NZr_h8E7ss)

2. F1 Score: EM과 다르게 부분 점수를 제공합니다. 예를 들어, 정답은 "Barack Obama"지만 예측이 "Obama"일 때, EM의 경우 0점을 받겠지만 F1 Score는 겹치는 단어도 있는 것을 고려해 부분 점수를 받을 수 있습니다.

![](https://lh6.googleusercontent.com/FpfHu0FtS5YiwGgSEFNAarBqT1IhMfMoDMKJXmwzz57RhsF7Bdra0H7TZQ6ugSk9L3aKzaSz_SFlQJtENxnhhq7cjx6NJFGMG8ytSDs6_8BNiBQ3YkIhZNSPW7-c7tZvr911cYH8rSp2pES9_5-zBUg)

EM 기준으로 리더보드 등수가 반영되고, F1은 참고용으로만 활용됩니다.

### Detailed Timeline

- 프로젝트 전체 기간 (3주) : 6월 5일 (월) 10:00 ~ 6월 22일 (목) 19:00

- 팀 병합 기간 : 6월 6일 (화) 16:00 까지

- 팀명 컨벤션 : 도메인팀번호(2자리)조 / ex) CV_03조, NLP_02조, RecSys_08조

- 리더보드 제출 오픈 : 6월 7일 (수) 10:00

- 리더보드 제출 마감 : 6월 22일 (목) 19:00

- 최종 리더보드 (Private) 공개 : 6월 22일 (목) 20:00

- GPU 서버 할당 : 6월 5일 (월) 10:00

- GPU 서버 회수 : 8월 18일 (금) 16:00

- 교육 종료(수료식) : 8월 2일 (수)

- 포스트세션(추가 GPU 제공 기간) : 8월 2일 (수) ~ 8월 18일 (금)

### Competiton Rule

- [대회 참여 제한] NLP 도메인을 수강하고 있는 캠퍼에 한하여 리더보드 제출이 가능합니다.

- [팀 결성 기간] 팀 결성은 대회 페이지 공개 후 2일차 오후 4시까지 필수로 진행해 주세요. 팀이 완전히 결성되기 전까지는 리더보드 제출이 불가합니다.

- [일일 제출횟수] 일일 제출횟수는 '팀 단위 10회'로 제한합니다. (일일횟수 초기화 자정 진행)

- [외부 데이터셋 규정] KLUE-MRC 데이터셋을 제외한 모든 외부 데이터 사용 허용합니다.

- [기학습 가중치 사용] 기학습 가중치는 제한 없이 모두 허용하나, KLUE MRC 데이터로 학습된 기학습 가중치 (pretrained weight) 사용은 금지합니다. 가중치는 모두 public 에 공개되어 있고 저작권 문제 없이 누구나 사용 가능해야 합니다. 사용하는 기학습 가중치는 공지 게시판의 ‘기학습 가중치 사용 공지’ 게시글에 댓글로 가중치 및 접근 가능한 링크를 반드시 공유합니다. 이미 공유되어 있을 경우 추가로 공유주실 필요는 없습니다.

- [평가 데이터 활용] 학습 효율 측면에서 테스트셋을 분석하고 사용(학습)하는 행위는 본 대회에서는 금지합니다. (눈으로 직접 판별 후 라벨링 하는 행위 포함)

- [데이터셋 저작권] 대회 데이터셋은 '캠프 교육용 라이선스' 아래 사용 가능합니다. 저작권 관련 세부 내용은 부스트코스 공지사항을 반드시 참고 해주세요.

---

AI Stages 대회 공통사항

- [Private Sharing 금지] 비공개적으로 다른 팀과 코드 혹은 데이터를 공유하는 것은 허용하지 않습니다.  
  코드 공유는 반드시 대회 게시판을 통해 공개적으로 진행되어야 합니다.

- [최종 결과 검증 절차] 리더보드 상위권 대상으로추후 코드 검수가 필요한 대상으로 판단될 경우 개별 연락을 통해 추가 검수 절차를 안내드릴 수 있습니다. 반드시 결과가 재현될 수 있도록 최종 코드를 정리 부탁드립니다. 부정행위가 의심될 경우에는 결과 재현을 요구할 수 있으며, 재현이 어려울 경우 리더보드 순위표에서 제외될 수 있습니다.

- [공유 문화] 공개적으로 토론 게시판을 통해 모델링에 대한 아이디어 혹은 작성한 코드를 공유하실 것을 권장 드립니다. 공유 문화를 통해서 더욱 뛰어난 모델을 대회 참가자 분들과 같이 개발해 보시길 바랍니다.

- [대회 참가 기본 매너] 좋은 대회 문화 정착을 위해 아래 명시된 행위는 지양합니다.

- 대회 종료를 앞두고 (3일 전) 높은 점수를 얻을 수 있는 전체 코드를 공유하는 행위

- 타 참가자와 토론이 아닌 단순 솔루션을 캐내는 행위

## 

### Data Composition

![](https://lh4.googleusercontent.com/XJappA1f8bDXu8DBg2rimLPadUxwE_P1bp3J2Bee1ECwTF67HR8fr2GxbAze5HGjFvvdtcLAyoyrS6AWE1Fh6UQUcTXmue196hcgchagAqQsAvPg0hu-zeWlQXRQnsnF19KVuLthlhJKaHESbDQwCkQ)

MRC 데이터의 경우, HuggingFace에서 제공하는 datasets 라이브러리를 이용하여 접근이 가능합니다. 해당 directory를 dataset_name 으로 저장한 후, 아래의 코드를 활용하여 불러올 수 있습니다.

#train_dataset을 불러오고 싶은 경우

from datasets import load_from_disk

dataset = load_from_disk("./data/train_dataset/")

print(dataset)

Retrieval 과정에서 사용하는 문서 집합(corpus)은 ./data/wikipedia_documents.json 으로 저장되어있습니다. 약 5만 7천개의 unique 한 문서로 이루어져 있습니다.

데이터셋은 편의성을 위해 Huggingface 에서 제공하는 datasets를 이용하여 pyarrow 형식의 데이터로 저장되어있습니다. 다음은 ./data 구조입니다.

#전체 데이터

./data/

    # 학습에 사용할 데이터셋. train 과 validation 으로 구성

    ./train_dataset/

    # 제출에 사용될 데이터셋. validation 으로 구성

    ./test_dataset/

    # 위키피디아 문서 집합. retrieval을 위해 쓰이는 corpus.

    ./wikipedia_documents.json

### Data Example

![](https://lh6.googleusercontent.com/7tBiqEIdiztXpBhUTAg4m9BlB7x2nxJ0jfqcO_UhTUJ6T6g-OfIEkzaNsL9Q1q4OMP8FnuRkNDIZlnv0SHSYkMtd2Fmdanb8i-KUMW-8C0qUMQuySNh27d0V75fzt2_eb80okV1bYDUhzkX28VNpDpI)

- id: 질문의 고유 id

- question: 질문

- answers: 답변에 대한 정보. 하나의 질문에 하나의 답변만 존재함

- answer_start : 답변의 시작 위치

- text: 답변의 텍스트

- context: 답변이 포함된 문서

- title: 문서의 제목

- document_id: 문서의 고유 id

![](https://lh4.googleusercontent.com/0Kp3WWgaT5aFg6fImu9a4pwva1bTIPWTNpOfy8HF0LSIcqozvfDIKisEy831EL8eVybm5inlSjrvuoI9GkDqxkJd-Don91NW_yyW85a1WQcIMcn-z8b8UwGqSPD1h9abDbcNOZC1lZNwrwZ5t_krNHo)

전체 600개 Test dataset 중에 리더보드에서 계산이 되는 Public data는 240개, 최종 리더보드의 최종 등수를 위해 계산되는 Private data는 360개로 구성됩니다.

[리더보드 공개용 평가 데이터 (Public)]

평가 데이터는 학습데이터와 거의 동일한 구조이고 리더보드 공개용 test dataset의 데이터에는 id 와 question 만 주어집니다. 즉, ODQA 전용이며 제출할 답안을 생성하기 위해서 ./data/wikipedia_documents.json 을 활용합니다.

[최종 평가(Private) 데이터]

대회가 종료된 후에는 캠퍼분들에게 제공되지 않았던 테스트 데이터를(Private data) 기반으로 최종 리더보드 등수가 결정됩니다.

### Directory Structure and File Description

.

|-- code

|   |-- arguments.py

|   |-- assets

|   |-- inference.py

|   |-- install

|   |-- README.md

|   |-- retrieval.py

|   |-- trainer_qa.py

|   |-- train.py

|   `-- utils_qa.py

`-- data

    |-- test_dataset

    |-- train_dataset

    `-- wikipedia_documents.json

---

- ./install/

- 요구사항 설치 파일

- retrieval.py

- sparse retreiver 모듈 제공

- arguments.py

- 실행되는 모든 argument 가 dataclass 의 형태로 저장되어있음

- trainer_qa.py

- MRC 모델 학습에 필요한 trainer 제공.

- utils_qa.py

- 기타 유틸 함수 제공

- train.py

- MRC, Retrieval 모델 학습 및 평가

- inference.py

- ODQA 모델 평가 또는 제출 파일 (predictions.json) 생성

### 설치 방법

---

#### 요구 사항

데이터 (약 51.2 MB)

tar -xzf data.tar.gz

필요한 파이썬 패키지 설치.

$ cd /opt/ml/input/code/install

$ conda init

$ . ~/.bashrc

(base) $ conda create -n mrc python=3.10 -y

(base) $ conda activate mrc

(mrc) $ chmod +x install_requirements.sh

(mrc) $ bash ./install_requirements.sh

### 훈련, 평가, 추론

---

#### 학습 방법

arguments 에 대한 세팅을 직접하고 싶다면 arguments.py 를 참고해주세요.

#### 학습 예시 (train_dataset 사용)

python train.py --output_dir ./models/train_dataset --do_train

#### 평가 방법

MRC 모델의 평가는(--do_eval) 따로 설정해야 합니다. 위 학습 예시에 단순히 --do_eval 을 추가로 입력해서 훈련 및 평가를 동시에 진행할 수도 있습니다.

### mrc 모델 평가 (train_dataset 사용)

python train.py --output_dir ./outputs/train_dataset --model_name_or_path ./models/train_dataset/ --do_eval

#### 추론 방법

retrieval 과 mrc 모델의 학습이 완료되면 inference.py 를 이용해 ODQA를 진행할 수 있습니다.

- 학습한 모델의 test_dataset에 대한 결과를 제출하기 위해선 추론(--do_predict)만 진행하면 됩니다.

- 학습한 모델이 train_dataset 대해서 ODQA 성능이 어떻게 나오는지 알고 싶다면 평가(--do_eval)를 진행하면 됩니다.

### ODQA 실행 (test_dataset 사용)

#### wandb 가 로그인 되어있다면 자동으로 결과가 wandb 에 저장됩니다. 아니면 단순히 출력됩니다

python inference.py --output_dir ./outputs/test_dataset/ --dataset_name ../data/test_dataset/ --model_name_or_path ./models/train_dataset/ --do_predict

### 제출 방법

---

inference.py 파일을 위 예시처럼 --do_predict 으로 실행하면 --output_dir 위치에 predictions.json 이라는 파일이 생성됩니다. 해당 파일을 제출해주시면 됩니다.

### 주의사항

---

1. train.py 에서 sparse embedding 을 훈련하고 저장하는 과정은 시간이 오래 걸리지 않아 따로 argument 의 default 가 True로 설정되어 있습니다. 실행 후 sparse_embedding.bin 과 tfidfv.bin 이 저장이 됩니다. 만약 sparse retrieval 관련 코드를 수정한다면, 꼭 두 파일을 지우고 다시 실행해주세요! 안그러면 존재하는 파일이 load 됩니다.

2. 모델의 경우 --overwrite_cache 를 추가하지 않으면 같은 디렉토리에 저장되지 않습니다.

3. ./outputs/디렉토리 또한 --overwrite_output_dir 을 추가하지않으면 같은 디렉토리에 저장되지 않으므로, 같은 디렉토리로 중복 실행 시 실행되지 않을 수 있습니다

### Links

Download Data Link

https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000242/data/data.tar.gz

Download Baseline Code Link

https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000242/data/code.tar.gz

## Leader Board

![](https://lh3.googleusercontent.com/u5It-x-vCgozlEX5ikxhSUa2xq5MO2rD2QSzQqspoi4EnLKaj3-SqGW1IAeIHQupgcYPoKJAD6NMtxpIgjO-EnPojv6pxWpBHPiBvH5uyx9rfX4Gon3VT_1yTp8P5-FbrxpMpe1O7ae0lbLvIuGJmLs)

![](https://lh6.googleusercontent.com/9zyPOJn4DK5HxcRHGBAyQKyc2maTLdYUgrli1yCZzFn5I-o0NndRjfvHfJUh23QudTBppoT2oHPQHqfUKfVoKeHigJ2nYIj3g9xU5Hp-YKoB7dZMNobVI3Yq3GVA3QrQzppPur85pbGyUBx_LN7Qj4Y)

## ETC

[공유] PyTorch Lightning으로 Customize하기

Posted by 1pha

2023.05.23.20:53

 

# Pytorch Lightning

이제 제법 PyTorch로 코드를 짜는 경험들을 많이 해보셨을텐데요, PyTorch 코드가 생각보다 정형화되어 있다는 것을 알 수 있을 것입니다. 전체적인 파이프라인을 돌이켜보면

- DataLoader 구성

- Model 구성

- Backpropagation + weight update

- 전체 코드 실행하는 스크립트

정도로 정리할 수 있습니다. 처음 공부할 때 몇 번은 이걸 밑바닥부터 짜는 게 재밌지만, 프로젝트를 진행할수록 이 코드를 다시 짜는 게 여간 일이 아닐 수가 없습니다. Task가 조금 바뀌거나 데이터가 변경되면 바꿔줘야할 코드가 많기 때문에 이전에 사용된 코드를 재사용하는 것도 어렵습니다. 이런 귀찮은 과정들을 어느 정도 정형화해둔 PyTorch code template (혹은 boilerplate)들이 많이 존재하는데요, 여기 위에 코드를 짜도 고려해야될 부분들이 있습니다. 예를 들어 mixed precision이나 multi-gpu 환경에 맞춰서 코드를 짜야할 부분들은 boilerplate가 있어도 여간 귀찮은 작업이 아닐 수 없습니다. 또한 중간중간 모델을 저장하는 부분들이나 gradient accumulation 같이 사소하지만 귀찮은 부분들 또한 구성해야 하는데 이 또한 귀찮습니다.

 다행히도 현재 자연어처리 task를 손쉽게 처리해줄 수 있는 Huggingface 라이브러리는 이 모든 귀찮은 작업들을 대신 해주고 있습니다. 몇 가지의 flag를 통해서 위의 기능들을 모두 쉽게 Trainer 하나로 구현 가능하다는 장점이 있습니다. 하지만 손쉬운 사용에는 대가가 따르는데요, 도대체 Trainer의 train이 무슨 기능을 하고 있는지, 세부적으로 내가 원하는 기능을 조절할 수는 없는지 등의 불편함도 따릅니다.

이 부분을 간편하게 사용할 수 있도록 구현해준 라이브러리가 있는데요, 바로 [PyTorch Lightning](https://www.pytorchlightning.ai/)입니다. 해당 라이브러리가 광고(?)하는 코드를 살펴보면 생각보다 간단한 pytorch_lightning.LightningModule로 한 번에 많은 부분들을 해결하는 것을 볼 수 있는데요, 저희 MRC에도 한 번 적용해보면 좋을 것 같습니다.

## 기본적인 사용법

혹시 scikit-learn을 사용해보신 적이 있나요? scikit-learn은 굉장히 많은 머신러닝 모델들을 포함하고 있는데요, 사용법이 굉장히 간단합니다. 몇 줄만으로 모델을 정의하고 fitting까지 마무리한 후에 테스트 데이터에 대해 추론하는 것까지 단 몇 줄이면 가능합니다. 물론 이는 모델의 구조나 모델을 훈련하는 방식이 정형화 되어 있기 때문에 아래처럼 쉽게 짤 수 있는 구조인데요, pytorch 모델도 이렇게 쉽게 짜면 굉장히 편할 것 같습니다.

```
from sklearn.ensemble import RandomForestClassifier

cls = RandomForestClassifier()

cls.fit(X, y)

cls.predict(X_test)
```

위의 scikit-learn 구조를 보면, 모델을 정의하고, .fit 메소드를 통해 학습시킨 후, .predict 메소드로 추론까지 마무리할 수 있습니다. PyTorch Lightning도 비슷한 구조를 지향하고 있는데요, 라이트닝의 메인홈페이지에 나오는 구조를 보시면 알 수 있습니다. 아래는 다른 소개예시에 나온 코드인데요.

![](https://lh6.googleusercontent.com/cdstmXeHNfea3HiuPr79LmNHUbXZNGmlY_kieuD611FzM2tP26B3jAzORjnTDPviCQy4WFOw7EKP4uHLSJbVM8_v03OPizqwfa7BVZUtu8KxAc4RL0A520WI-gWxfGUCahTrSXrvKXMopVVcZ3_0b3Q)

살펴보면, pytorchlightning을 통한 model class하나를 생성해서, 해당 클래스 내부에 훈련과 추론하는 코드를 작성하면, 그 외의 부분은 lightning 내부에서 자동으로 처리해주는 구조입니다. 그 후 trainer를 정의하여 방금 정의한 모델과 사용자가 필요한 DataLoader를 넣어서 .fit만 해주면 간단해집니다. 이 .fit은 Huggingface Trainer의 .train과 유사한 기능을 합니다. 그 대신 .train 내에서 돌아가는 코드를 우리가 직접 구현할 수 있다는 것에서 차이가 있습니다.

그렇다면 모델은 어떻게 구성해줘야할까요? 이걸 자세하게 구성해주는 방식은 많은데, 가장 간단한 방식은 batch 단위 행동을 정의해주면 됩니다. batch 단위의 행동이란, dataloader에 대해 for loop를 돌릴 때 내부에서 정의되는 동작입니다. 이를 *_step이라는 메소드를 오버라이드하여 사용할 수 있습니다. 다음과 같이 사용할 수 있습니다. 바로 아래와 같이요

```
import pytorch_lightning as pl

import transformers as tfm

class MRCLightning(pl.LightningModule):

    def __init__(

        self,

        model,

        args,

        train_dataset=None,

        eval_dataset=None,

        eval_examples=None,

        post_process_function: callable = None,

        compute_metrics: callable = None

    ):

        self.model = model

        self.args = args

        self.train_dataset = train_dataset

        self.eval_dataset = eval_dataset

        self.eval_examples = eval_examples

        self.post_process_function = post_process_function

        self.compute_metrics = compute_metrics

    def configure_optimizers(self):

        # <https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L1058>

        decay_parameters = tfm.trainer_pt_utils.get_parameter_names(

            self.model, tfm.pytorch_utils.ALL_LAYERNORM_LAYERS

        )

        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        optimizer_grouped_parameters = [

            {

                "params": [

                    p for n, p in self.model.named_parameters() if n in decay_parameters

                ],

                "weight_decay": self.args.weight_decay,

            },

            {

                "params": [

                    p

                    for n, p in self.model.named_parameters()

                    if n not in decay_parameters

                ],

                "weight_decay": 0.0,

            },

        ]

        optimizer_kwargs = {

            "lr": self.args.learning_rate,

            "betas": (self.args.adam_beta1, self.args.adam_beta2),

            "eps": self.args.adam_epsilon,

        }

        self.optimizer = tfm.AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

    def training_step(self, batch):

        qa_output = self.model(**batch)

        return {"loss": qa_output["loss"]}

    def validation_step(self, batch):

        """여기에 Validation Step을 구성해보세요.

        trainer_qa.py의 QuestionAnsweringTrainer의 evaluate, predict 메소드를 참고하세요.

        혹은 `run_qa_no_trainer.py` 예제를 참고하셔도 좋습니다.

        <https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_no_trainer.py>

        """

        raise NotImplemented

"""

Baseline의 train_dataset, data_collator를 가져와서

train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=data_collator)

"""

model = MRCLightning(args)

wandb_logger = pl.loggers.WandbLogger(project="mrc")

model_ckpt = pl.callbacks.ModelCheckpoint(

    dirpath=args.output_dir,

    monitor="validate_acc",

    filename="{epoch}-{val_loss:.2f}",

    mode="max",

    save_top_k=3,

)

trainer = pl.Trainer(

    gpus=1,

    precision=16,

    logger=wandb_logger,

    max_epochs=args.n_epochs,

    gradient_clip_val=args.clip_grad,

    callbacks=[

        model_ckpt,

    ],

)

trainer.fit(model, train_loader)
```

기존 Huggingface Trainer는 내부 행동 정의가 힘든 편인데, 여기서는 여러분 마음대로 forward도 정의하고 loss를 계산하는 방식 같은 여러 가지 작업들을 직접 조작할 수 있게 됩니다. 여기서 loss만 return하도록 맞춰주면, 그 외의 귀찮은 optimizer.step(), optimizer.zero_grad()를 적어주지 않아도 됩니다. 코드의 양이 줄어든다는 것은 단순히 가독성을 위함이 아니라, 사용자가 만들어낼 수 있는 에러들의 경우의 수를 낮춰주는 좋은 역할도 있습니다. 하지만 이렇게 래핑된 라이브러리들은 디버깅을 할 때 생각보다 애를 먹을 수 있습니다. 내가 생각한 로직이 내부에서 의도한대로 돌아가지 않는 경우들이 있기 때문에 이를 언제나 고려해야합니다.

Huggingface trainer에서는 이것이 불가능하지는 않습니다. 저희가 베이스로 사용하는 [Bert 모델](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1831)을 살펴보면 아래처럼 구성되어 있는데요

```
def forward(

        self,

        input_ids: Optional[torch.Tensor] = None,

        attention_mask: Optional[torch.Tensor] = None,

        token_type_ids: Optional[torch.Tensor] = None,

        position_ids: Optional[torch.Tensor] = None,

        head_mask: Optional[torch.Tensor] = None,

        inputs_embeds: Optional[torch.Tensor] = None,

        start_positions: Optional[torch.Tensor] = None,

        end_positions: Optional[torch.Tensor] = None,

        output_attentions: Optional[bool] = None,

        output_hidden_states: Optional[bool] = None,

        return_dict: Optional[bool] = None,

    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:

        r"""

        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):

            Labels for position (index) of the start of the labelled span for computing the token classification loss.

            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence

            are not taken into account for computing the loss.

        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):

            Labels for position (index) of the end of the labelled span for computing the token classification loss.

            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence

            are not taken into account for computing the loss.

        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(

            input_ids,

            attention_mask=attention_mask,

            token_type_ids=token_type_ids,

            position_ids=position_ids,

            head_mask=head_mask,

            inputs_embeds=inputs_embeds,

            output_attentions=output_attentions,

            output_hidden_states=output_hidden_states,

            return_dict=return_dict,

        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1).contiguous()

        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None

        if start_positions is not None and end_positions is not None:

#If we are on multi-GPU, split add a dimensionif len(start_positions.size()) > 1:

                start_positions = start_positions.squeeze(-1)

            if len(end_positions.size()) > 1:

                end_positions = end_positions.squeeze(-1)

#sometimes the start/end positions are outside our model inputs, we ignore these terms

            ignored_index = start_logits.size(1)

            start_positions = start_positions.clamp(0, ignored_index)

            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            start_loss = loss_fct(start_logits, start_positions)

            end_loss = loss_fct(end_logits, end_positions)

            total_loss = (start_loss + end_loss) / 2

        if not return_dict:

            output = (start_logits, end_logits) + outputs[2:]

            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(

            loss=total_loss,

            start_logits=start_logits,

            end_logits=end_logits,

            hidden_states=outputs.hidden_states,

            attentions=outputs.attentions,

        )
```

이 클래스를 상속 받아서 내부 행동 정의를 원하는대로 정리해준다면, 충분히 Lightning을 쓰지 않고도 모델을 건드릴 수 있게 됩니다.

## 그 외 유용한 기능들

### Trainer Arguments

제일 좋은 건 PyTorch lightning의 [공식 documentation](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html#pytorch_lightning.trainer.trainer.Trainer)을 확인해보는 것입니다. 그 중 몇 가지를 살펴보겠습니다.

- gradient_clip_val Gradient clipping을 걸어줄 수 있습니다. gradientclipalgorithm 또한 설정가능합니다.

- accumulate_grad_batches Gradient accumulation을 별도로 구현하지 않고 이를 간편하게 argument로 조정할 수 있습니다.

- fast_dev_run 종종 디버깅을 하고 싶은데 데이터가 너무 많아서 한세월 걸리는 경우들이 있습니다. 이럴 때 전체 데이터를 살펴보지 않고 일부 데이터만 살펴보도록 정해줄 수 있는 방법입니다.

- resume_from_checkpoint 저장한 체크포인트를 다시 불러와서 학습할 때 쓸 수 있습니다. 이 또한 구현하는 것이 생각보다 귀찮은데 간편하게 사용할 수 있습니다. 이는 Huggingface의 구현된 것과 비슷한 기능을 합니다.

- precision Mixed precision을 구현해서 모델이 차지하는 메모리를 줄여줍니다. 매번 불러와서 쓰는 것이 귀찮은데 이 또한 쉽게 쓸 수 있습니다.

### Callbacks

콜백함수는 간단하게 설명하면 train step이나 validate step 전후에 추가적인 기능을 수행하고 싶을 때 사용할 수 있습니다. 대표적으로, 모델 체크포인트를 저장하는 [ModelCheckpoint](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html) 콜백함수가 있는데요, 이를 사용하면 특정 epoch이나 step 전후에 언제 모델을 저장할지 지정해줄 수 있습니다.

[EarlyStopping](https://pytorch-lightning.readthedocs.io/en/stable/common/early_stopping.html) 또한 콜백함수를 통해 쉽게 걸어줄 수 있습니다.

### Loggers

Metric들을 더 쉽게 시각화해주는 라이브러리들이 많습니다. MLflow나 Wandb 같은 것들이 그 예인데, 방금 예제에서 살펴본 self.log를 라이트닝이 지원하는 사용자가 원하는 라이브러리에 기록하는 것이 가능합니다. ([공식 documentation 참고](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html))

## 마치며

PyTorch Lightning은 생각보다 회사에서도 간단하게 mock-up을 실행해봐야할 때 종종 쓰이는 라이브러리입니다. 과거 0.x 버전에는 잔 버그가 너무 많아서 사실 테스트하는 과정에서 에러 수정하는 일이 더 잦았기 때문에 많이 쓰이지 않았지만, 현재 1.7+ 버전까지 등장한 이후로는 에러가 많이 줄어서 사용하기 굉장히 간편한 라이브러리입니다. 하지만 몇 가지 단점이 있는데, 그 중 하나가 multi-gpu 환경에서 DDP(Distributed Data Parallel)가 제대로 작동하지 않는 이슈입니다. 실제로 Pytorch Lightning github에 방문해보면 Issue에서 comment 기준으로 sort 해봤을 때 상위에 DDP 관련된 이슈가 많은 것을 확인할 수 있습니다. 간단하게 테스트해보고, 해당 태스크가 자세히 실험해보거나 더 큰 스케일로 올려서 실험해봐야 한다고 판단이 되면, 직접 trainer를 짜거나 dependency 없이 짜는 것을 추천드립니다. 그리고 자연어 업계에서는 Huggingface 사용이 훨씬 압도적으로 많기 때문에, 당연히 Huggingface에 더 익숙해지는 것을 추천합니다!  

[공유] Huggingface Space에서 프로젝트 참고하기

Posted by 1pha

2023.05.23.20:54

 

# HuggingFace Space

저는 종종 개발하다보면, 제 모델이 방구석 여포가 되는 것을 보면서 마음이 아팠는데요, 아무래도 AI 개발자다보니 이를 직접 웹에 배포하는 일은 아무래도 부담이 되고 시간이 뺏기지 않을 수 없습니다. 어디에 내 모델을 배포하는 데에는 크게 두 가지 과정을 수반해야 합니다.

- 모델을 웹 환경에서 제공하는 일

- 웹을 띄울 수 있는 서버 관리

이 두 과정 모두 귀찮은 일입니다. 하지만 기어코 세상에 내야겠다면... Huggingface Space를 참고해보시면 좋을 것 같아 소개합니다. 여기에는 Huggingface에서 제공하는 간단한 방법을 통해 여러분들의 모델을 업로드하고, 다른 유저들에게 공유할 수 있는 공간입니다. 최근에는 다양한 비전 모델도 Huggingface에서 제공하다보니, 비전 관련된 모델들도 여기에 전시해둘 수 있습니다.

![](https://lh5.googleusercontent.com/DmDMR8-w6L6RLTFmKDVpmAGz2rvVBGKtqhyPJqPrmSFnuNWnqNMAQD6ftGDA5Kzx31w57FGPoUrHybjSK56BdnfINWxfQ2UT5UcYjgGojXtRxEkC89tU4jDvszv3NSvFdmFlCL9GOiKWpKUkeEL-rHU)

대회 중에 뜬금 없이 Space가 등장한 이유는 무엇일까요? 여기서 여러 프로젝트들을 둘러보면서 괜찮은 QA 모델이 있다면, 어떻게 train을 했는지 찾아갈 수 있는 source가 제공되기 때문입니다. 하나의 space에 들어간 후 모델의 소스를 제공해주는 경우는 우상단에 Linked Models를 눌러서 어떻게 학습되었는지 살펴볼 수 있습니다. 학습에 더 이상 얻을 정보가 없다면.. 여기라도 뒤져보는 것을 추천합니다. Question Answering을 검색해보면 생각보다 검색결과가 아주 많습니다. 이 중 다른 언어로 train된 모델이 많은데, 제가 발견한 바로 아직까지 한국어 QA 모델은 누군가 업로드 하지 않은 것 같네요

![](https://lh6.googleusercontent.com/O7BoVEJK6ocGsicbiR_NqUipKmKcij5gu5zJmNQOKcWFbUbqdIepHblIvR-bft0_jAMISPfZQNba_pRgNxDrNifbHgfu2jpBl99NVRcOySgkG4YhKvP8VjAkj33E3B3UdhIXh6jl173vJwmc4eVjxo8)하나의 space를 들어가봤는데요, 생각보다 검색결과가 나쁘지 않다는 것을 알 수 있었습니다.

![](https://lh3.googleusercontent.com/c8GprINws-Mv3E8JeLPfcwORQfe5EidbQqgpfzibjWyi3ZsA7fp9Gz_LAcJ-aIbJNIlWRsLhxtsZm07PbLi4Ia6BtHzEQySYJ9l15o_bToZ6oTHh6QfwydVILHw1XWW1DbkCnWGGpOrrOwJuoVcx4Yw)

이 Space는 어떤 모델을 참고했을까요? 우상단의 Linked models를 눌러봅시다.

![](https://lh4.googleusercontent.com/Nom_EGGOVU2dB2V_TSBlt_vkb8h0a1j8e2-BHG1yoHS2-gKze3TyVFsM4s5Jt7w3Fd0-hQwP4muU8m5t2J3xR0ObRoefrUAuq5umCBEuPYhWYmsC_SBEFNnwwgbPYJCdCgnIWXQ8xc-GJzjlkESAgHE)

첫 번째로 참조되어 있는 모델을 들어가서 확인해보면 Long-form Question Answering이라는 데이터셋에 학습이된 BERT라는 것을 알 수 있었습니다.

![](https://lh6.googleusercontent.com/F9_78IQT1mhfSrvoN-0kky0CKPmgCi8cs9fiIXLXyRcyA3U48HrxLRQ1T2P9SH0hqVDJ9x1bzWOlnIhCggF0gb0Z98GUzeMEKZupm4upa4CraTLDMBLEy7YZ1jaQ_JSdUYIEsfYrw5xRuYVqZtb5vTg)

이렇게 찾다보면 우리의 목적에 부합하는 space를 찾아서 새로운 insight를 얻어볼 수 있을 것 같습니다. Space에 대한 구체적인 정보는 [Documentation](https://huggingface.co/docs/hub/spaces)에 정리되어 있습니다.

Back to List

[공유] lint, testing을 pre-commit으로 수행해보기

Posted by 1pha

2023.05.23.20:58

협업에서 동일한 code convention은 생산성을 높이는데 매우 중요합니다. 협업이 진행되면 Merge conflict, PR review 등 여러 상황에서 팀원의 코드를 읽고 이해하는 상황이 발생합니다. 이 때, 팀원의 코딩 습관이 나와 맞지 않거나, 혹은 코드 작성 규칙이 나와 다르다면 생산성이 저하됩니다.

캠퍼분들도 이러한 상황을 막고자 flake8, black 등을 활용하여 이미 통일된 코드 컨벤션으로 협업을 수행하고 계실겁니다. 본 토론글에서는 이러한 코드 컨벤션 통일을 자동화하는 간단한 방법을 소개해드리고자 합니다.

## Pre-commit

Hook은 Git에서 특정 상황에 특정 스크립트를 실행하는 기능입니다. 여기서 ‘특정 상황’이라고 하는 것은 commit, push, checkout 등 git에서 발생하는 여러 액션들이 해당됩니다.

본 섹션에서는 commit 등록 직전에 실행되는 hook인 pre-commit을 활용해서 간단한 linter를 실행시키겠습니다.

.pre-commit-config.yaml 파일을 project root directory에 생성 후, 아래 yaml 파일의 내용을 기입해봅시다.

touch .pre-commit-config.yaml

# .pre-commit-config.yaml

repos:

-   repo: <https://github.com/pre-commit/pre-commit-hooks>

    rev: v4.4.0

    hooks:

    -   id: trailing-whitespace

    -   id: end-of-file-fixer

    -   id: check-yaml

    -   id: check-added-large-files

-   repo: <https://github.com/psf/black>

    rev: 22.10.0

    hooks:

    -   id: black

# -   repo: <https://github.com/pycqa/flake8>

# rev: 6.0.0

# hooks:

# - id: flake8

-   repo: <https://github.com/pycqa/isort>

    rev: 5.10.1

    hooks:

    -   id: isort

        files: "\\\\\\\\.(py)$"

해당 pre-commit에서 사용할 모듈은 black과 isort입니다. black은 파이썬 표준 규약을 활용해서 code convention을 통일시켜주는 도구입니다. isort는 python script의 import 구문들을 정렬, 정리해주는 도구입니다. 해당 두 가지 tool 모두 사용자가 원하는 규약들만 적용해주는 기능을 지원합니다. e.g., line당 max-length는 체크하지 않겠다.

.pre-commit-config.yaml에서 hooks의 arguments로 해당 옵션들을 추가해줄 수도 있습니다.

또한 flake8을 추가해서 statistic code analysis를 수행할 수도 있습니다. 사용되지 않은 불필요한 변수와 라이브러리, 불필요한 코드 구문들을 코드 레벨 단계에서 분석해서 testing을 수행하는 도구입니다. 해당 도구를 사용하면 MRC baseline의 많은 부분들을 수정해야 하기 때문에, 주석 처리해뒀습니다. 혹시 엄격한 testing을 원하시는 캠퍼들은 해당 주석을 제거 후 사용하시면 됩니다.

python module dependency를 통해서 pre-commit을 사용하기 위해 아래 명령어를 실행합니다.

pip install pre-commit

테스트를 위해 간단한 스크립트를 아래와 같이 추가합니다. 본 섹션에서 추가하는 커밋은 이후에 지울 것입니다. 또한 아래의 예제 스크립트가 아니라 다른 어떠한 파이썬 파일을 추가하셔서 테스트 하셔도 무방합니다.

# test.py

```
from transformers import BartTokenizer, BartModel

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

model = BartModel.from_pretrained('facebook/bart-base')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
```

해당 파일을 stage에 올린 후 commit을 합니다.

```
git add test.py

git commit -m "Test pre-commit"
```

실행 후에는 커밋이 되지 않고 아래와 같이 변경사항들이 표시 됩니다.

![](https://lh4.googleusercontent.com/9v9ojk8Zr0WAe8X-9RWBmE6wLpGylPzlBhtaWwbpNJS_RnrAo15ljdH_2-xgzfHUDCF3YZzr5SWl_f7W4wSaXPMW2n2LW7OQyDd2HWXGT48VzXnC-QKZiSF3KCV9to4rXNNgqFtfZZydepH9jWrgWPA)

black과 isort에서 변경사항이 발생했고, pre-commit 내의 test를 통과하지 못해서 commit이 되지 않았음을 알 수 있습니다. 자동으로 변경사항들이 파일에 적용됐기 때문에 다시 stage에 파일을 올리고 commit을 수행합니다.

git add test.py

git commit -m "Test pre-commit"

![](https://lh4.googleusercontent.com/uMcIR7G97dYzsGkGebFJzWXceJFUM41r_TqlsUWR29fD--rcqVWs3zGeFWGqdDOyrQIj1syo_rejjfb1WN81f3AfM0EHEgqdkpZQsnv-lL3afveiOVBd3WurDUyrzj-_d8AmSDlAq06MEZ0RuhzMrkk)

성공적으로 commit이 local git에 등록된 것을 알 수 있습니다.

## Next..

- flake8을 팀 내에서 사용하기로 결정했다면, project repository의 Github Action에서 flake8을 testing에 등록하는 것도 좋습니다. e.g., Github Action의 testing을 통과한 코드만을 PR에서 merge 할 수 있도록 관리하고 싶다.

- black, flake8, isort 모두 여러 오픈소스 프로젝트에서 사용되는 도구들입니다. 관심 있으신 오픈소스 프로젝트의 pre-commit configurations를 따라해보는 것도 좋은 방법입니다.

[공유] Data Augmentation using Question Generation

Posted by 1pha

2023.05.23.21:00

 

# Data Augmentation

Data Augmentation이란 기본적으로 학습 가능한 데이터를 변형을 통해 추가함으로써 사용가능한 pair 수를 늘려 모델이 보다 다양하고 많은 수의 데이터를 학습할 수 있도록하는 기법입니다. 모델은 data augmentation을 통해 학습하는 범위가 더 늘어나기 때문에 보다 강건한 학습을 통해 성능 개선을 이뤄낼 수 있습니다. 이번에는 Context, Question, Answer가 하나의 pair를 이루는 데이터 형태에서 사용할 수 있는 data augmentation 방법인 qQuestion gGeneration을 소개해드리고자 합니다.

# Question Generation

이번 토론글에서는 아주 간단한 question generation 방법을 소개하고자 합니다. 이미 구글링을 통해 1, 2, 3기 캠퍼 분들의 솔루션이나 literature review를 통해 question generation을 접하신 분들도 많으실겁니다. 혹시 더 좋고, 효율적인 방법들을 아신다면 토론 게시판에 공유해주시면 감사하겠습니다!

오늘 제시되는 방법론은 부스트캠프 A.I. Tech 2기에 캠퍼로 참가하셨던 박성호 조교님께서 공유해주신 방법입니다. 해당 방법은 아래 레포를 기반으로 시작되었습니다.

[GitHub - codertimo/KorQuAD-Question-Generation: question generation model with KorQuAD dataset](https://github.com/codertimo/KorQuAD-Question-Generation)

위 repository는 Question Generation을 위한 Dataset을 사용했습니다. 즉, Context, Answer를 통해서 Question을 생성하는데, 외부 데이터 사용이 불가능한 MRC 대회에서는 사용이 불가능한 방법이었습니다.

해당 대회에 맞춰 사용하기 위해 Wikipedia data를 사용하기로 했습니다. (Title, Context)로 구성되어있는 Wikipedia data에서 title을 answer로 간주하는 것입니다. 왜냐하면 title은 context의 전반적인 내용을 함축하고 있는 구문이라고 치부할 수 있기 때문입니다.

다만, 지엽적인 질문에 대해서는 title이 context의 내용을 담고 있다고 할 수 없습니다. 글의 말미에 이러한 처리에 대한 고민과 미봉책을 남겨두겠습니다.

위 repository의 scripts/run_finetune.py를 대회에서 직접 사용할 수 없기 때문에 아래와 같이 코드를 수정해서 사용했습니다. 해당 코드의 링크 또한 첨부했습니다.

https://github.com/naem1023/mrc-level2-nlp-09/blob/main/question_generation/question_generation.py

```
import sys

from pandas.core.indexes.base import ensure_index

sys.path.append("..")

import random

from argparse import ArgumentParser

import pandas as pd

import torch

from tokenizers import SentencePieceBPETokenizer

from torch.utils.data import DataLoader

from tqdm import tqdm

from transformers import GPT2LMHeadModel

from QG.korquad_qg.config import QGConfig

from QG.korquad_qg.dataset import (MAX_QUESTION_SPACE, MIN_QUESTION_SPACE, QGDecodingDataset, load_wiki_dataset)

model = GPT2LMHeadModel.from_pretrained("taeminlee/kogpt2")

model.load_state_dict(torch.load('QG/model/QG_kogpt2.pth', map_location="cpu"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)

tokenizer = SentencePieceBPETokenizer.from_file(

    vocab_filename="QG/tokenizer/vocab.json", merges_filename="QG/tokenizer/merges.txt", add_prefix_space=False

)

examples_list = load_wiki_dataset('QG_utils/remove_wiki_text_title.csv')

random.shuffle(examples_list)

examples=[]

d_id=[]

for i in examples_list:

    examples.append(i[0])

    d_id.append(i[1])

dataset = QGDecodingDataset(examples, tokenizer, 512)

dataloader = DataLoader(dataset, batch_size=1)

model = model.to(device)

model.eval()

generated_results = []

for i, batch in tqdm(enumerate(dataloader), desc="generate", total=len(dataloader)):

    input_ids, attention_mask = (v.to(device) for v in batch)

    origin_seq_len = input_ids.size(-1)

    decoded_sequences = model.generate(

        input_ids=input_ids,

        attention_mask=attention_mask,

        max_length=origin_seq_len + MAX_QUESTION_SPACE,

        min_length=origin_seq_len + MIN_QUESTION_SPACE,

        pad_token_id=0,

        bos_token_id=1,

        eos_token_id=2,

        num_beams=5,

        repetition_penalty=1.3,

        no_repeat_ngram_size=3,

        num_return_sequences=1,

    )

    for decoded_tokens in decoded_sequences.tolist():

        decoded_question_text = tokenizer.decode(decoded_tokens[origin_seq_len:])

        decoded_question_text = decoded_question_text.split("</s>")[0].replace("<s>", "")

        generated_results.append(

            (i, examples[i].answer, examples[i].question, decoded_question_text, d_id[i])

        )

with open('question_generation_id.tsv', "w") as f:

    for context, answer, question, generated_question,d_id in generated_results:

        f.write(f"{generated_question}\\\\t{answer}\\\\t{d_id}\\\\n")
```

해당 코드는 바로 실행시키실 수 없고, 몇가지 설정 사항이 필요합니다. question_generation/README.md에 사용 방법을 첨부해뒀으니 해당 순서대로 시도해보시면 됩니다.

오래된 코드이기 때문에 개선 가능한 사항들이 있습니다.

- 보다 더 큰 vocabulary를 활용한 tokenizer 사용하기

- SKT KoGPT2 외에 더 좋은 생성 모델 사용하기

- maxlength, minlength, num_beams 등의 generation strategy와 관련된 hyperparameter들을 조정해서 대회에 fitting되는 값 탐색

- mix precision, fp16 model 등을 활용해서 보다 빠르게 question generation 수행해보기

## 문제점

Title이 Context에 대한 Answer라고 단정지은 방법론이고 이를 생성을 통해 해결하고자 하기 때문에 여러 문제점들이 있습니다.

- Context의 내용을 반복하는 형태의 Question이 생성될 수 있습니다.

- Question에 Answer(Title)이 포함될 수 있습니다. 문제에 따라서 다르겠지만, 저희 팀은 이러한 케이스를 좋은 generation 결과라고 간주하지 않았습니다.

- Title과 연관된 Question만을 생성할 수 밖에 없습니다.

- e.g., “대한민국”에 대한 Wikipedia context에는 지리, 기후, 동식물 등 여러가지 정보가 내포돼있고 이에 대한 Question-Answer pair를 생성하기도 해야합니다. 하지만 Title인 “대한민국”을 Answer라고 간주한 방법론이기 때문에 지리, 기후, 동식물과 관련된 Answer를 가지는 Question은 생성할 수 없습니다.

첫 번째와 두 번째 문제는 간단한 후처리나 앞서 NLP 강의에서 배웠던 내용들을 통해서 필터링 할 수 있습니다. 세 번째 문제의 경우 여러가지 관점을 가질 수 있습니다. 단순히 generation data의 양을 줄이는 것도 방법입니다. 혹은 보다 좋은 question generation 방법론을 고안하여 한계를 원론적인 한계를 극복하는 것도 방법입니다.

이러한 문제를 해결하는 방법들은 팀의 점수 향상에 있어서 핵심적인 역할을 합니다. 따라서, 토론 게시판에 업로드하기 어려울 수도 있다고 생각합니다. 그래도 여러 캠퍼 분들이 함께 성장한다는 측면에서 원론적인 고민점들이나 간단한 해결방법들을 공유하는 것이 본인의 성장에도 도움이 된다고 생각합니다.

여러 캠퍼 분들의 정보 공유를 기대하겠습니다!
