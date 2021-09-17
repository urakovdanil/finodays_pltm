import pickle
from random import choice
import os
import time
import json
import datetime
import sys
import joblib
import spacy

import messages


def predict(message_text, classifier_eng_vote, ln_eng, mb_eng, bnb_eng):
    preds = classifier_eng_vote.predict([message_text])[0]
    if preds == 0:
        score = (ln_eng.predict_proba([message_text])[0][0]+mb_eng.predict_proba([message_text])[0][0]+bnb_eng.predict_proba([message_text])[0][0])/3
    else:
        score = (ln_eng.predict_proba([message_text])[0][1]+mb_eng.predict_proba([message_text])[0][1]+bnb_eng.predict_proba([message_text])[0][1])/3
    loaded_model = spacy.load("model_artifacts")
    parsed_text = loaded_model(message_text)
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Положительный отзыв"
        score_spacy = parsed_text.cats["pos"]
    else:
        prediction = "Негативный отзыв"
        score_spacy = parsed_text.cats["neg"]
    return preds, score, prediction, score_spacy


def predict_rus(message_text, classifier_rus_vote, ln_rus, mb_rus, bnb_rus):
    preds = classifier_rus_vote.predict([message_text])[0]
    if preds == 0:
        score = (ln_rus.predict_proba([message_text])[0][0]+mb_rus.predict_proba([message_text])[0][0]+bnb_rus.predict_proba([message_text])[0][0])/3
    else:
        score = (ln_rus.predict_proba([message_text])[0][1]+mb_rus.predict_proba([message_text])[0][1]+bnb_rus.predict_proba([message_text])[0][1])/3
    loaded_model = spacy.load("model_artifacts")
    parsed_text = loaded_model(message_text)
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Положительный отзыв"
        score_spacy = parsed_text.cats["pos"]
    else:
        prediction = "Негативный отзыв"
        score_spacy = parsed_text.cats["neg"]
    return preds, score, prediction, score_spacy


def message_process_eng(classifier_eng_vote, ln_eng, mb_eng, bnb_eng):
    start_time = datetime.datetime.today().replace(microsecond=0)
    cur_path_txt = str(round(100 * time.time())) + '.txt'
    cur_path_json = str(round(100 * time.time())) + '.json'
    with open(cur_path_txt, 'w', encoding='cp1250') as cur_message:
        cur_key = choice(list(messages.test_messages_eng.keys()))
        cur_message.write(messages.test_messages_eng[cur_key])

    with open(cur_path_txt, 'r', encoding='cp1250') as f:
        content = '\n'.join(f.readlines())

    p1, p2, p3, p4 = map(str, predict(content, classifier_eng_vote, ln_eng, mb_eng, bnb_eng))

    data = {
        'timestamp': str(start_time),
        'content': content,
        'ratings': {
            'reply': '\n'.join([p1, p2, p3, p4]),
        },
        'recommendations': 'Специалист по жопе'
    }
    with open(cur_path_json, 'w') as write_file:
        json.dump(data, write_file)

    try:
        os.remove(cur_path_txt)
    except:
        pass

    reply = '\n'.join([f'Сообщение получено в {data["timestamp"]}',
                       f'Текст: {data["content"]}',
                       f'Предсказанная тональность: {data["ratings"]["reply"]}',
                       f'Рекомендации: {data["recommendations"]}'])
    return reply


def message_process_rus(classifier_rus_vote, ln_rus, mb_rus, bnb_rus):
    start_time = datetime.datetime.today().replace(microsecond=0)
    cur_path_txt = str(round(100 * time.time())) + '.txt'
    cur_path_json = str(round(100 * time.time())) + '.json'
    with open(cur_path_txt, 'w', encoding='utf-8') as cur_message:
        cur_key = choice(list(messages.test_messages_rus.keys()))
        cur_message.write(messages.test_messages_rus[cur_key])

    with open(cur_path_txt, 'r', encoding='utf-8') as f:
        content = '\n'.join(f.readlines())

    p1, p2, p3, p4 = map(str, predict(content, classifier_rus_vote, ln_rus, mb_rus, bnb_rus))

    data = {
        'timestamp': str(start_time),
        'content': content,
        'ratings': {
            'reply': '\n'.join([p1, p2, p3, p4]),
        },
        'recommendations': 'Специалист по жопе'
    }
    with open(cur_path_json, 'w') as write_file:
        json.dump(data, write_file)

    try:
        os.remove(cur_path_txt)
    except:
        pass

    reply = '\n'.join([f'Сообщение получено в {data["timestamp"]}',
                       f'Текст: {data["content"]}',
                       f'Предсказанная тональность: {data["ratings"]["reply"]}',
                       f'Рекомендации: {data["recommendations"]}'])
    return reply


def message_process_main(message):
    start_time = datetime.datetime.today().replace(microsecond=0)

    model_linear_svc = pickle.load(open('models/LinearSVC.pkl', 'rb'))
    model_logistic_regression = pickle.load(open('models/LogisticRegression.pkl', 'rb'))
    model_sgd = pickle.load(open('models/SGDClassifier.pkl', 'rb'))

    cur_path_txt = str(round(100 * time.time())) + '.txt'
    cur_path_json = str(round(100 * time.time())) + '.json'

    with open(cur_path_txt, 'w', encoding='cp1250') as cur_message:
        cur_message.write(message.text)

    with open(cur_path_txt, 'r', encoding='cp1250') as f:
        model_logistic_regression_prediction = model_logistic_regression.predict_proba(f)[-1][-1]
    with open(cur_path_txt, 'r', encoding='cp1250') as f:
        model_linear_svc_prediction = model_linear_svc.predict(f)
    with open(cur_path_txt, 'r', encoding='cp1250') as f:
        model_sgd_prediction = model_sgd.predict_proba(f)[-1][-1]

    with open(cur_path_txt, 'r', encoding='cp1250') as f:
        content = '\n'.join(f.readlines())

    data = {
        'timestamp': str(start_time),
        'content': content,
        'ratings': {
            'logistic_regression': round(float(model_logistic_regression_prediction), 4),
            'linear_svc': round(float(model_linear_svc_prediction), 4),
            'sgd': round(float(model_sgd_prediction), 4),
        },
        'recommendations': 'Специалист по жопе'
    }
    with open(cur_path_json, 'w') as write_file:
        json.dump(data, write_file)

    os.remove(cur_path_txt)

    reply = '\n'.join([f'Сообщение получено в {data["timestamp"]}',
                       f'Текст: {data["content"]}',
                       f'Предсказанная тональность: {data["ratings"]["logistic_regression"]}',
                       f'Предсказанная тональность: {data["ratings"]["linear_svc"]}',
                       f'Предсказанная тональность: {data["ratings"]["sgd"]}',
                       f'Рекомендации: {data["recommendations"]}'])
    return reply


print(sys.version)
