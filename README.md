# 패혈증 원인균 분류 인공지능
본 프로젝트는 인공지능을 이용하여 패혈증 원인균을 찾아내는 방법에 관한 내용이다. 패혈증 원인균을 이용한 수학적인 특징을 나타내기 위해 미생물 연료전지를 활용해 혈액 속 미생물이 만들어 내는 전위값을 측정하게 된다. 이를 이용해 인공지능을 학습시키고 원인균을 분류하게 된다.
## 데이터 수집
Maltose,	Galactose,	Saccharose, D-Mannitol, Lactose, Dextrose, Maltose,	Galactose,	Saccharose,	D-Mannitol,	Lactose,	Dextrose등의 당류를 이용해서 혈액 속에 포함된 미생물의 연료전지 반응을 측정합니다.

 혈액 속의 미생물의 양, 온도 등에 의해서 전반적인 전압 값이 변할 수 있으므로 Saccharose를 기준으로 다른 값을 상대적인 값으로 측정합니다.
미생물의 종류로는 Escherichia coli, Staphylococcus aureus, bacillus pumilus 3개의 종을 시범적으로 학습시킵니다. 

## 모델 구성
모델은 입력층, 은닉층, 출력층으로 구성합니다. 활성화함수로는 ReLU 함수를 사용하며, 오차함수로는 Cross Entropy, 출력층 활성화 함수로는 Softmax 함수를 이용합니다. 배치처리를 이용하여 학습 속도를 향상시킵니다.

## 학습
테스트 데이터와 훈련 데이터를 분리하여 학습시키고 오차 값을 점검합니다. 
