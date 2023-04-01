---
layout: archive
title: "chapter 3. classification"
categories:
  - ML
use_math: true
---

## Chapter 3: 분류

<br>3.1 MNIST
----------------------
데이터셋: MNIST  
<br>
MNIST 데이터셋 불러오기  

```py
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()
```
> dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])    

- 딕셔너리 구조
    - DESCR: 데이터셋을 설명하는 키
    - data: 샘플이 하나의 행, 특성이 하나의 열로 구성된 배열을 가진 키
    - target: 레이블 배열을 담은 키

<br>

```py
X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)
```
> (70000, 784)  
> (70000,)  
<br>
- 70000개의 이미지가 있다.
- 각 이미지는 28 x 28 =784개의 특성이 있다. (28 x 28 픽셀)
- 개개의 특성은 0(흰색)부터 255(검은색)까지의 픽셀 강도를 나타낸다.  

<br>

```py
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")

save_fig("some_digit_plot")
plt.show()
```
![image](https://user-images.githubusercontent.com/108905986/229296067-3afb89de-d238-42fc-88a3-c8e00d2dfb53.png)

- X[0]: 70000개의 이미지 중에서 하나만 살펴보자!
- 이미지의 모양을 28 x 28로 바꿔야 한다.
- 요약: 샘플의 특성 벡터를 추출해서 28 x 28 배열로 크기를 바꾸고 imshow() 함수를 써서 그리면 된다.

<br>

```py
y[0]
```
> ‘5’  
<br>
- y = mnist["target"]
- 위의 그림이 5로 보이기는 한데… 진짜 5일까? → 확인해야 함
- 따라서 y[0]을 실행한 결과 ‘5’가 맞다.
- 그런데 문자열이네..??

<br>

```py
y = y.astype(np.uint8)
y[0]
```
> 5  

- 정수형으로 바꼈다. astype() 사용
- 대부분 머신러닝 알고리즘은 숫자를 사용한다.

<br>

```py
plt.figure(figsize=(9,9))
example_images = X[:100]
plot_digits(example_images, images_per_row=10)
save_fig("more_digits_plot")
plt.show()
```
(일부 코드는 생략)  
![image](https://user-images.githubusercontent.com/108905986/229296231-6811bf3b-4daf-4cfa-829e-6384df8c3703.png)  
- MNIST의 이미지 샘플 중 100개만 확인해보기

<br>

```py
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```
- 데이터를 자세히 조사하기 전에 항상 훈련 / 테스트 세트로 분리해야 한다.
- 훈련 세트: 앞쪽 60000개
- 테스트 세트: 뒤쪽 10000개
- 어떤 학습 알고리즘은 훈련 샘플의 순서에 민감해서 많은 비슷한 샘플이 연이어 나타나면 성능이 나빠질 수도 있다. → 데이터 셋을 섞으면 이런 문제를 해결할 수 있다.


<br>
<br>

3.2 이진 분류기 훈련  
----------------------
- 5만 식별해보자.
- 이진 분류기: 5가 맞다 / 5가 아니다

<br>

```py
y_train_5= (y_train== 5)
y_test_5= (y_test== 5)
```
- 타깃 벡터 생성

<br>

```py
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)
```
- 분류 모델: SGDClassifier
- SGD = 확률적 경사 하강법(stochastic Gradient Descent)
- 장점: 매우 큰 데이터셋을 효율적으로 처리한다.
- tol: 줄어드는 loss값이 tol값보다 클 동안만 반복학습을 진행한다. 일정 순간이 되면 loss값의 변화가 미세하게 줄어들기 때문에 시간이 절약된다.

<br>

```py
sgd_clf.predict([some_digit])
```
> array([ True])  
<br>
- 숫자 5의 이미지를 감지해보기
- 모델을 통해 X[0]이 5인지 예측해보면, 맞음(True)

<br>
<br>

3.3 성능 측정  
----------------------
<mark style='background-color: #ffdce0'> 중요!! </mark>

<br>

**3.3.1 교차 검증을 사용한 정확도(Accuracy) 측정**
- 교차 검증을 왜 하는데?: overfitting을 막기 위해서
- <mark style='background-color: #fff5b1'> 분류에서 Accuracy를 성능 측정 지표로 잘 안쓴다. </mark>

```py
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

# shuffle=False가 기본값이기 때문에 random_state를 삭제하던지 shuffle=True로 지정하라는 경고가 발생한다.
skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)  #fold를 3개로 나눈다.
```

```py
for train_index, test_index in skfolds.split(X_train, y_train_5): 
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index] #훈련 폴드
    y_train_folds = y_train_5[train_index] #훈련 폴드
    X_test_fold = X_train[test_index] #테스트 폴드
    y_test_fold = y_train_5[test_index] #테스트 폴드

    clone_clf.fit(X_train_folds, y_train_folds) #훈련 폴드로 훈련하기
    y_pred = clone_clf.predict(X_test_fold) #테스트 폴드로 예측하기
    n_correct = sum(y_pred == y_test_fold) #올바른 예측 수 세기
    print(n_correct / len(y_pred)) #평균, 정확한 예측 비율 출력
```
> 0.9669  
> 0.91625  
> 0.96785  

- StratifiedKFold 사용
- 분류에서 성능 측정 지표로 자주 쓴다.
- KFold를 보완한 것이다.
    - KFold 단점: 0, 1, 2가 정답이다. 만약 Fold가 3이라고 하자. 0과 1만 가지고 2를 예측할 수 없고, 1과 2만 가지고 0을 예측할 수 없고, 0과 2만 가지고 1을 예측할 수 없다.
- StratifiedKFold: **불균형한 분포도를 가진 레이블 데이터 집합을 위한 KFold 방식이다.** 특정 레이블 값이 특이하게 많거나, 매우 적어서 값의 분포가 한쪽으로 치우치는 경우에 적합하다.
- skfolds.split(X_train, y_train_5) → KFold와는 다르게 StratifiedKFold는 레이블 데이터 분포도에 따라 학습 / 검증 데이터를 나누기 때문에 split()의 인자로 feature data(X_train) 뿐만 아니라 레이블 데이터(y_train_5)를 넣어야 한다.
- clone_clf = clone(sgd_clf) → 매 시행마다 분류기 객체 복제
- 훈련 폴드로 훈련시키고 테스트 폴드로 예측한다.
- n_correct = sum(y_pred == y_test_fold) → 올바른 예측 수 세기
- print(n_correct / len(y_pred)) → 평균을 구한 것으로 정확한 예측 비율 출력

<br>

```py
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv = 3, scoring = 'accuracy')
```
> array([0.95035, 0.96035, 0.9604 ])  
<br>
- KFold를 사용해도 결과는 비슷하다.

<br>

```py
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator): #나만의 변환기 생성
    def fit(self, X, y=None): #fit은 X와 y 매개변수만 갖는다. 
        pass
    def predict(self, X): #X만 매개변수로 갖는다.
        return np.zeros((len(X), 1), dtype=bool) #모두 False로
```
```py
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```
> array([0.91125, 0.90855, 0.90915])  

- **모든 이미지를 ‘5 아님’ 클래스로 분류**하는 더미 분류기 생성
- 나만의 변환기 만들기!!
- BaseEstimator: 파이프라인과 그리드 탐색에 필요한 _get_params()과 set_params() 메서드를 사용할 수 있도록 지원한다.
- def fit(self, X, y=None) → 이 함수는 더미라서 아무것도 안하고 그냥 패스한다.
- def predict(self, X) → 입력값으로 들어오는 X 데이터셋의 크기만큼 모두 0으로 만들고, bool이라서 모두 False가 된다. 죽, 모두 5가 아님(False)을 나타내는 분류기인 것이다.
- **다 5가 아니라고 측정했는데 정확도가 90%가 나온다고….??? → 불균형 데이터의 경우 왜곡된 결과가 나온다!!!!**
- <span style="color:orange">**정확도(Accuracy)를 분류기의 성능 측정 지표로 잘 쓰지 않는다.**</span> 왜냐하면 <span style="color:orange">**불균형한 데이터셋**</span>을 다룰 수도 있기 때문이다.
- 참고: [https://woochan-autobiography.tistory.com/820](https://woochan-autobiography.tistory.com/820)
- 참고:[https://yganalyst.github.io/ml/ML_chap2/](https://yganalyst.github.io/ml/ML_chap2/)


<br>
<br>

**3.3.2 오차 행렬(Confusion matrix)**

- zmffotm A의 샘플이 클래스 B로 분류된 횟수를 세는 것이다.
- 예를 들어 분류기가 숫자 5의 이미지를 3으로 잘못 분류한 횟수를 알고 싶다면? → 오차 행렬의 5행 3열을 보면 된다.

<br>
