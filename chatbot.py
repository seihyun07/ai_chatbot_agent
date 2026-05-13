# chatbot.py
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"[경고] GPU 메모리 할당 오류: {e}")

lemmatizer = WordNetLemmatizer()

try:
    with open('intents.json', 'r', encoding='utf-8') as file:
        intents = json.load(file)
except FileNotFoundError:
    raise Exception("오류: 'intents.json' 파일을 찾을 수 없습니다.")

try:
    with open('words.pkl', 'rb') as f:
        words = pickle.load(f)
    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)
    model = tf.keras.models.load_model('chatbot_model.keras')
except Exception as e:
    raise Exception(f"종속 파일 로드 오류: {e}")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]), verbose=0)[0]
    
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "제가 아직 깊이 학습하지 못한 세부 주제인 것 같습니다. 조금 더 큰 단원명으로 말씀해 주시거나 다른 키워드로 질문해 주세요!"
        
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
            
    return "해당 내용을 찾을 수 없습니다."

if __name__ == "__main__":
    print("="*60)
    print("과목별 학습 도우미 챗봇이 시작되었습니다.")
    print("세션을 종료하려면 'quit' 또는 'exit'을 입력하세요.")
    print("="*60)
    
    valid_subjects = ['물리', '화학', '미적분', '영어', '화법과 작문', '프로그래밍', '기하']
    
    while True:
        # 1단계: 과목 입력
        print(f"\nBot  >> 어떤 과목이 어려우신가요? (지원 과목: {', '.join(valid_subjects)})")
        subject = input("User >> ").strip()
        
        if subject.lower() in ["quit", "exit"]:
            print("Bot  >> 학습 세션을 종료합니다. 수고하셨습니다!")
            break
            
        if subject not in valid_subjects:
            print("Bot  >> 죄송합니다. 현재 괄호 안의 명시된 과목들만 지원하고 있습니다. 정확한 과목명을 입력해 주세요.")
            continue
            
        # 2단계: 세부 주제 입력
        print(f"Bot  >> '{subject}' 과목을 선택하셨군요! 구체적으로 어떤 개념이나 단원이 이해하기 힘드신가요?")
        topic = input("User >> ").strip()
        
        if topic.lower() in ["quit", "exit"]:
            print("Bot  >> 학습 세션을 종료합니다. 수고하셨습니다!")
            break
            
        # 3단계: 과목명과 세부 주제를 결합하여 모델 입력값 생성 (예: "미적분 극한")
        combined_message = f"{subject} {topic}"
        
        try:
            ints = predict_class(combined_message, model)
            res = get_response(ints, intents)
            print(f"Bot  >> {res}")
        except Exception as e:
            print(f"Bot  >> [오류] 답변을 준비하는 중 문제가 발생했습니다: {e}")