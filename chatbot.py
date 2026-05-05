# chatbot.py
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import os

import nltk
from nltk.stem import WordNetLemmatizer

# Fixed (TensorFlow): GPU 런타임 환경에서 텐서플로우 엔진이 모든 VRAM을 
# 사전 점유하여 백그라운드 프로세스와 충돌하는 현상(OOM)을 방지하기 위한 메모리 동적 할당 로직
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"[경고] GPU 메모리 동적 할당 설정 중 내부 오류 발생: {e}")

lemmatizer = WordNetLemmatizer()

# Fixed: 훈련 스크립트와 동일한 UTF-8 인코딩 방식 적용을 통해 일관성 보장
try:
    with open('intents.json', 'r', encoding='utf-8') as file:
        intents = json.load(file)
except FileNotFoundError:
    raise Exception("오류: 'intents.json' 파일을 작업 디렉토리에서 찾을 수 없습니다.")

try:
    # 훈련 단계에서 직렬화된 어휘 사전 및 클래스 차원 메타데이터 로드
    with open('words.pkl', 'rb') as f:
        words = pickle.load(f)
    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)
    
    # Fixed (TensorFlow): 구버전 레거시 h5 포맷 대신 새로운 압축 아카이브 포맷인 
    #.keras 모델을 안전하게 로드하여 'Unknown activation function' 오류 회피
    model = tf.keras.models.load_model('chatbot_model.keras')
except Exception as e:
    raise Exception(f"종속 파일 로드 오류. 훈련 스크립트(training.py)가 정상적으로 완료되었는지 확인하십시오: {e}")

def clean_up_sentence(sentence):
    """사용자의 원시 발화 텍스트를 NLTK 엔진을 통해 토큰화하고 표제어를 추출하여 정규화하는 함수"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    """입력된 문장을 전체 어휘 사전과 대조하여 단어 가방(Bag of Words) 밀집 벡터로 변환하는 함수"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"어휘 사전 매칭 발견: {w}")
                    
    # 모델 입력 차원인 (1, Vocabulary Size) 형태에 맞추기 위해 Numpy 다차원 배열로 반환
    return np.array(bag)

def predict_class(sentence, model):
    """순방향 연산(Forward Pass)을 통해 문장의 의도 확률 분포를 추론하고 노이즈를 필터링하는 함수"""
    p = bow(sentence, words)
    
    # 수정 1: 결과가 [[...]] 형태로 나오므로 [0]을 붙여서 알맹이 리스트만 꺼냅니다.
    res = model.predict(np.array([p]), verbose=0)[0]
    
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # 수정 2: 확률값(x[1])을 기준으로 내림차순 정렬해야 합니다.
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        # 수정 3: r은 [인덱스, 확률] 형태의 리스트이므로 r[0], r[1]로 접근해야 합니다.
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        
    return return_list

def get_response(intents_list, intents_json):
    """분류된 최고 확률 의도를 바탕으로 JSON 지식 베이스에서 무작위 응답을 매핑하여 반환하는 함수"""
    if not intents_list:
        return "죄송합니다, 의도를 정확히 파악하지 못했습니다. 다시 말씀해 주시겠어요?"
        
    # 수정 4: intents_list는 리스트이므로 첫 번째 항목([0])의 'intent'를 가져와야 합니다.
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
            
    return result

# 콘솔 기반 실시간 대화형 인터페이스 루프 진입점
if __name__ == "__main__":
    print("="*60)
    print("AI 챗봇 신경망 런타임 초기화 완료. 터미널 대화를 시작합니다.")
    print("세션을 종료하려면 'quit' 또는 'exit'을 입력하세요.")
    print("="*60)
    
    while True:
        message = input("User >> ")
        if message.lower() in ["quit", "exit"]:
            print("Bot  >> 시스템 통신을 종료합니다. 이용해 주셔서 감사합니다.")
            break
            
        try:
            ints = predict_class(message, model)
            res = get_response(ints, intents)
            print(f"Bot  >> {res}")
        except Exception as e:
            print(f"Bot  >> [연산 오류 발생] 입력 텐서 처리 중 문제가 발생했습니다: {e}")