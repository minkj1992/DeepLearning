## Stemming and lemmatization with nltk


Word2Vec으로 문장 분류하기
=========

[참고블로그](https://ratsgo.github.io/natural%20language%20processing/2017/03/08/word2vec/)

+ `코퍼스(corpus, 말뭉치)`

+ `임베딩` = 단어를 벡터로 바꿔주는 방법

+ W2V's Feature
	+ 단어를 벡터화할 때 단어의 문맥적 의미를 보존한다.
	+ 예를들어 Man&Woman의 거리와 King&Queen 끼리의 거리가 유사하다.(유클리디안 거리 or 코사인 유사도를 통해 거리 측정 가능)

+ W2V 순서
	+ 1) 자료 수집(크롤링)
	+ 2) 포스태깅(형태소분석==토크나이징), 대표적인 한국어 포스태거(KoNLPy, cohension tokenizer[코퍼스의 출현빈도로 학습한 결과를 토대로 토큰을 나눠줌], 굳이 w2v에 품사정보까지 넣을 필요없다.)
	+ 3) W2V방법론 적용(with gensim)

	+ 3-1) w2v 작업환경 설정하기

```python

# Word2Vec embedding
from gensim.models import Word2Vec
embedding_model = Word2Vec(tokenized_contents, size=100, window = 2, min_count=50, workers=4, iter=100, sg=1)

# window = 2 // 앞뒤로 두개까지 보라
# min_count = 50 // window 2중에서, 코퍼스 내 출현 빈도가 50번 미만인 단어는 분석에서 제외
# workers = 4 // cpu 쿼드 코어 사용하라
# iter = 100 //100번 반복학습
# sg = 1 // CBOW와 Skip-Gram 중 후자를 선택하라.
```
	+ 3-2) 유사도 측정(거리 계산)

> gensim 패키지가 제공하는 기능 중에 ‘most_similar’라는 함수가 있습니다. 두 벡터 사이의 코사인 유사도를 구해줍니다. 그 값이 작을 수록 비슷한 단어라는 뜻인데, 아래 코드는 ‘디자인’이라는 단어와 가장 비슷한(코사인 유사도가 큰) 100개 단어를 출력하라는 지시입니다.

```python
# check embedding result
print(embedding_model.most_similar(positive=["디자인"], topn=100))
```

	+4) 단어벡터로 가중치 행렬 만들기
> 문맥적 정보가 보존된 상태의 단어 벡터 사이의 거리(유사도)를 구하고 이를 가중치 삼아 각 문장별로 스코어를 구한다. 이렇게 구한 스코어를 바탕으로 각 리뷰 문장을 특정 기능에 할당(분류)한다.

Word2Vec의 아웃풋은 아래와 같은 단어벡터 행렬입니다. 첫번째 열의 각 요소는 위의 코드 가운데 ‘min_count’ 조건을 만족하는 코퍼스 내 모든 단어가 됩니다.
아래 행렬의 열의 개수는 ‘‘임베딩 차원수(size) + 1(단어가 속한 열)’이 됩니다. 다시 말해 행벡터가 각 단어에 대응하는 단어벡터라는 뜻이지요. 하지만 행렬 각 요소의 숫자들은 사람이 이해하기 어렵습니다. 


`거리행렬(distance matrix)`

한편 예시에서는 ‘배터리’와 ‘발열’ 사이의 거리가 1, ‘배터리’와 ‘은’은 10으로 나왔습니다. 그렇다면 ‘은’보다는 ‘발열’이 ‘배터리’와 유사한 단어라고 볼 수 있겠네요. 이것이 이제 우리가 만들 가중치 행렬이 지향하는 핵심 원리입니다. 즉, 특정 쿼리 단어(예를 들어 ‘배터리’)와 거리가 가까운(=의미가 유사한) 단어는 높은 가중치, 그렇지 않은 단어는 낮은 가중치를 가지도록 가중치행렬을 만들어보자는 것입니다. 이를 수식으로 나타내면 다음과 같습니다.


거리행렬의 모든 대각성분은 0이 됩니다.
