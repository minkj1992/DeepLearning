Word2vec 관련 이론 
===========

> 예를 들어, ‘한국’에 대한 벡터에서 ‘서울’에 대한 벡터를 빼고, ‘도쿄’ 에 대한 벡터를 넣는다면? 이 벡터 연산 결과를 통해 나온 새로운 벡터와 가장 가까운 단어를 찾으면, 놀랍게도 이 방식으로 ‘일본’ 이라는 결과를 얻을 수가 있다

+ word2vec
+GloVe방법

## 1. How to Train Word Vectors?

> "모든 Word Embedding 관련 학습들은 기본적으로 언어학의 ‘Distributional Hypothesis‘ 라는 가정에 입각하여 이루어진다.  이는, ‘비슷한 분포를 가진 단어들은 비슷한 의미를 가진다’ 라는 의미이다. 여기서 비슷한 분포를 가졌다는 것은 기본적으로 단어들이 같은 문맥에서 등장한다는 것을 의미한다."

+ NNLM(w2v 이전)
[그림]("https://shuuki4.files.wordpress.com/2016/01/nnlm.png")

+ RNNLM(w2v 이전)
[그림]("https://shuuki4.files.wordpress.com/2016/01/rnnlm.png")

NNLM을 Recurrent Neural Netwok의 형태로 변형한 것이다.
이 네트워크는 기본적으로 Projection Layer 없이 Input, Hidden, Output Layer로만 구성되는 대신, Hidden Layer에 Recurrent한 연결이 있어 이전 시간의 Hidden Layer의 입력이 다시 입력되는 형식으로 구성되어 있다.


## 2. Word2Vec
> Continuous Word Embedding 학습 모형 CBOW(Continuous Bag-of-Words) +  SKIP-gram

+ 1) CBOW 
"주어진 단어에 대해 앞 뒤로 C/2개 씩 총 C개의 단어를 Input으로 사용하여, 주어진 단어를 맞추기 위한 네트워크를 만든다."
CBOW 모델은 크게 Input Layer, Projection Layer, Output Layer로 이루어져 있다. 그림에는 중간 레이어가 Hidden Layer라고 표시되어 있기는 하지만, Input에서 중간 레이어로 가는 과정이 weight를 곱해주는 것이라기 보다는 단순히 Projection하는 과정에 가까우므로 Projection Layer라는 이름이 더 적절할 것 같다. 

Input에서는 NNLM 모델과 똑같이 단어를 one-hot encoding으로 넣어주고, 여러 개의 단어를 각각 projection 시킨 후 그 벡터들의 평균을 구해서 Projection Layer에 보낸다. 그 뒤는 여기에 Weight Matrix를 곱해서 Output Layer로 보내고 softmax 계산을 한 후, 이 결과를 진짜 단어의 one-hot encoding과 비교하여 에러를 계산한다.


+ 2) Skip-gram
> "Skip-gram 모델은 CBOW와는 반대 방향의 모델이라고 생각할 수 있을 것 같다. 현재 주어진 단어 하나를 가지고 주위에 등장하는 나머지 몇 가지의 단어들의 등장 여부를 유추하는 것이다."

이 때 예측하는 단어들의 경우 현재 단어 주위에서 샘플링하는데, ‘가까이 위치해있는 단어일 수록 현재 단어와 관련이 더 많은 단어일 것이다’ 라는 생각을 적용하기 위해 멀리 떨어져있는 단어일수록 낮은 확률로 택하는 방법을 사용한다. 나머지 구조는 CBOW와 방향만 반대일 뿐 굉장히 유사하다.


## 3. V to ln(v): Complexity Reduction

> "영어 단어의 총 개수는 백만개가 넘는다고 한다. 그런데 네트워크의 Output Layer에서 Softmax 계산을 하기 위해서는 각 단어에 대해 전부 계산을 해서 normalization을 해주어야 하고, 이에 따라 추가적인 연산이 엄청나게 늘어나서 대부분의 연산을 이 부분에 쏟아야 한다. 이를 방지하기 위해서 이 부분의 계산량을 줄이는 방법들이 개발되었는데, Hierarchical Softmax와 Negative Sampling이 그 방법들이다."

+ 1) Hierarchical Softmax
"Hierarchical Softmax는 계산량이 많은 Softmax function 대신 보다 빠르게 계산가능한 multinomial distribution function을 사용하는 테크닉이다. " 
이 방법에서는 각 단어들을 leaves로 가지는 binary tree를 하나 만들어놓은 다음(complete 할 필요는 없지만, full 할 필요는 있을 것 같다), 해당하는 단어의 확률을 계산할 때 root에서부터 해당 leaf로 가는 path에 따라서 확률을 곱해나가는 식으로 해당 단어가 나올 최종적인 확률을 계산한다.

+ 2) Negative Sampling
> "Negative Sampling은 Hierarchial Softmax의 대체재로 사용할 수 있는 방법이다. 전체적인 아이디어는 ‘Softmax에서 너무 많은 단어들에 대해 계산을 해야하니, 여기서 몇 개만 샘플링해서 계산하면 안될까?’ 라는 생각에서 시작된다. 전체 단어들에 대해 계산을 하는 대신, 그 중에서 일부만 뽑아서 softmax 계산을 하고 normalization을 해주는 것이다."

"이 때 실제 target으로 사용하는 단어의 경우 반드시 계산을 해야하므로 이를 ‘positive sample’ 이라고 부르고, 나머지 ‘negative sample’ 들을 어떻게 뽑느냐가 문제가 된다. 이 뽑는 방법을 어떻게 결정하느냐에 따라 Negative sampling의 성능도 달라지고, 이는 보통 실험적으로 결정한다."

Error Function : "https://shuuki4.files.wordpress.com/2016/01/nsequation.png"

기본적으로 보고있는 단어 w와 목표로 하는 단어 c를 뽑아서 (w,c)로 둔 후, positive sample의 경우 ‘이 (w,c) 조합이 이 corpus에 있을 확률’ 을 정의하고, negative sample의 경우 ‘이 (w,c) 조합이 이 corpus에 없을 확률’을 정의하여 각각을 더하고 log를 취해서 정리하면 위과 같은 형태의 식이 나온다.

## 5. Further Method: Subsampling Frequent Words

## 6. Performance Comparison

## 결론

