# training.py
import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

# Fixed: 최신 NLTK 환경에서 요구하는 필수 언어 리소스를 명시적으로 일괄 다운로드
# punkt_tab 및 omw-1.4 누락으로 인한 런타임 LookupError 원천 차단
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) 
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

# Fixed: 데이터 로드 시 인코딩 강제 지정 (다국어 환경 문자 깨짐 및 JSONDecodeError 방지)
try:
    with open('intents.json', 'r', encoding='utf-8') as file:
        intents = json.load(file)
except FileNotFoundError:
    raise Exception("오류: 'intents.json' 파일이 존재하지 않습니다. 작업 디렉토리 구조를 확인하세요.")

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',', '을', '를', '이', '가']

# JSON 구조 파싱 및 자연어 토큰화
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # 단어 수준 토큰화 (문장을 개별 형태소 단위로 분할)
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # 발화(pattern)와 태그(intent)를 튜플 구조로 매핑하여 문서 집합에 보존
        documents.append((word_list, intent['tag']))
        
        # 고유한 의도(클래스) 수집
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# 단어의 원형(표제어) 추출, 소문자화, 중복 제거를 통한 어휘 사전(Vocabulary) 정밀화
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Fixed: 챗봇 추론 환경(chatbot.py)에서 텐서 구조를 동일하게 재현할 수 있도록
# 메타데이터(어휘 사전 및 클래스 차원 정보)를 피클 바이너리로 동기화 저장
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# 딥러닝 입력 계층 주입을 위한 벡터 데이터 생성 (단어 가방 및 원-핫 인코딩)
training_data_x = []
training_data_y = []

# 원-핫 인코딩(One-Hot Encoding)을 위한 영벡터(Zero Vector) 사전 할당
output_empty =  [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # 전체 어휘 사전을 순회하며 단어의 존재 여부를 1과 0의 이진 값으로 벡터화
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        
    output_row = list(output_empty)
    # 해당 문서가 속한 특정 클래스 인덱스만 1로 활성화 (소프트맥스 학습용 라벨)
    output_row[classes.index(document[1])] = 1
    
    # Fixed: 레거시 코드의 np.array(training) 일괄 변환 시 발생하는 
    # Numpy Ragged Array(불규칙 차원 배열) 경고 및 오류를 회피하기 위해 x와 y를 분리 적재
    training_data_x.append(bag)
    training_data_y.append(output_row)

# 데이터 편향을 막기 위해 Feature와 Label 쌍을 유지하며 스토캐스틱(Stochastic) 무작위 셔플링
combined = list(zip(training_data_x, training_data_y))
random.shuffle(combined)
training_data_x, training_data_y = zip(*combined)

# 머신러닝 연산에 최적화된 Numpy 고정 다차원 배열로 안전하게 타입 캐스팅
train_x = np.array(training_data_x)
train_y = np.array(training_data_y)

print(f"텐서 네트워크 Shape 검증 - Input 차원: {train_x.shape}, Output 차원: {train_y.shape}")

# Fixed (TensorFlow): Keras 순차적 다층 퍼셉트론(Sequential MLP) 아키텍처 재구성
model = tf.keras.Sequential()
# Keras 최신 규격에 맞게 명시적인 Input 레이어 추가 (특성 개수인 13차원 지정)
model.add(tf.keras.layers.Input(shape=(len(train_x[0]),)))
# 은닉층 1: 128 노드, ReLU 활성화 함수
model.add(tf.keras.layers.Dense(128, activation='relu'))
# 과적합 방지를 위한 드롭아웃 (50% 노드 비활성화)
model.add(tf.keras.layers.Dropout(0.5))
# 은닉층 2: 64 노드, 비선형성 추가 추출
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
# 출력층: 전체 의도(Class) 개수와 동일한 노드 수, 확률 분포 정규화를 위한 Softmax 활성화
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

# Fixed (TensorFlow): 최신 Keras API에서 폐기된 'lr'과 'decay' 매개변수를 완전히 제거하고 
# 최신 표준 규격인 'learning_rate' 인자를 사용하여 ValueError 컴파일 충돌 방지
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print("인공신경망 가중치 훈련 루프 시작...")
# verbose=1 옵션을 통해 훈련 진행도를 에포크 단위로 터미널에 렌더링
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Fixed (TensorFlow): TF 2.15+ 환경에서 역직렬화 커스텀 파싱 오류가 발생하는 레거시 H5 포맷을 폐기하고 
# 모든 토폴로지, 가중치, 옵티마이저 메타데이터를 압축 저장하는 최신 Keras V3 규격 '.keras' 포맷으로 저장
model.save('chatbot_model.keras')
print("모델 훈련 루프 종료. 'chatbot_model.keras' 파일 저장 성공.")