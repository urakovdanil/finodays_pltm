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


def predict(message_text, classifier_eng_vote, ln_eng, mb_eng, bnb_eng, spacy_model):
    preds = classifier_eng_vote.predict([message_text])[0]
    if preds == 0:
        score = (ln_eng.predict_proba([message_text])[0][0]+mb_eng.predict_proba([message_text])[0][0]+bnb_eng.predict_proba([message_text])[0][0])/3
    else:
        score = (ln_eng.predict_proba([message_text])[0][1]+mb_eng.predict_proba([message_text])[0][1]+bnb_eng.predict_proba([message_text])[0][1])/3
    loaded_model = spacy_model
    parsed_text = loaded_model(message_text)
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Положительный отзыв"
        score_spacy = parsed_text.cats["pos"]
    else:
        prediction = "Негативный отзыв" 
        score_spacy = parsed_text.cats["neg"]
        
    if score_spacy < 0.68:
        prediction = "Нейтральный отзыв"
    if score < 0.68:
        preds = "Нейтральный отзыв"
    return preds, score, prediction, score_spacy


def predict_rus(message_text, classifier_rus_vote, ln_rus, mb_rus, bnb_rus, spacy_model_rus):
    preds = classifier_rus_vote.predict([message_text])[0]
    if preds == 0:
        score = (ln_rus.predict_proba([message_text])[0][0]+mb_rus.predict_proba([message_text])[0][0]+bnb_rus.predict_proba([message_text])[0][0])/3
        preds = "Негативный отзыв"
    else:
        score = (ln_rus.predict_proba([message_text])[0][1]+mb_rus.predict_proba([message_text])[0][1]+bnb_rus.predict_proba([message_text])[0][1])/3
        preds = "Положительный отзыв"
    loaded_model = spacy_model_rus
    parsed_text = loaded_model(message_text)
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Положительный отзыв"
        score_spacy = parsed_text.cats["pos"]
    else:
        prediction = "Негативный отзыв" 
        score_spacy = parsed_text.cats["neg"]
        
    if score_spacy < 0.68:
        prediction = "Нейтральный отзыв"
    if score < 0.68:
        preds = "Нейтральный отзыв"
    return preds, score, prediction, score_spacy


def message_process_eng(classifier_eng_vote, ln_eng, mb_eng, bnb_eng, spacy_model):
    start_time = datetime.datetime.today().replace(microsecond=0)
    cur_path_txt = str(round(100 * time.time())) + '.txt'
    cur_path_json = str(round(100 * time.time())) + '.json'
    with open(cur_path_txt, 'w', encoding='cp1250') as cur_message:
        cur_key = choice(list(messages.test_messages_eng.keys()))
        cur_message.write(messages.test_messages_eng[cur_key])

    with open(cur_path_txt, 'r', encoding='cp1250') as f:
        content = '\n'.join(f.readlines())

    p1, p2, p3, p4 = predict(content, classifier_eng_vote, ln_eng, mb_eng, bnb_eng, spacy_model)

    specialist, recommendation = recommend(p1, p2)

    data = {
        'timestamp': str(start_time),
        'content': content,
        'ratings': {
            'vote_tone': str(p1),
            'vote_proba': str(p2), 
            'neuro_tone': str(p3), 
            'neuro_proba': str(p4)
        },
        'spec': specialist, 
        'recommendation': recommendation
    }
    with open(cur_path_json, 'w') as write_file:
        json.dump(data, write_file)

    try:
        os.remove(cur_path_txt)
    except:
        pass

    reply = '\n\n'.join([f'Сообщение получено в {data["timestamp"]} (UTC)',
                       f'Текст: {data["content"]}',
                       f'Предсказанная тональность по версии голосующей системы: {data["ratings"]["vote_tone"]}',
                       f'Степень уверенности в версии голосующей системы: {data["ratings"]["vote_proba"]}',
                       f'Предсказанная тональность по версии нейросети: {data["ratings"]["neuro_tone"]}',
                       f'Степень уверенности в версии нейросети: {data["ratings"]["neuro_proba"]}',
                       f'Рекомендованный специалист: {data["spec"]}',
                       f'Пояснения: {data["recommendation"]}'])
    return reply


def message_process_rus(classifier_rus_vote, ln_rus, mb_rus, bnb_rus, spacy_model_rus):
    start_time = datetime.datetime.today().replace(microsecond=0)
    cur_path_txt = str(round(100 * time.time())) + '.txt'
    cur_path_json = str(round(100 * time.time())) + '.json'
    with open(cur_path_txt, 'w', encoding='utf-8') as cur_message:
        cur_key = choice(list(messages.test_messages_rus.keys()))
        cur_message.write(messages.test_messages_rus[cur_key])

    with open(cur_path_txt, 'r', encoding='utf-8') as f:
        content = '\n'.join(f.readlines())

    p1, p2, p3, p4 = predict_rus(content, classifier_rus_vote, ln_rus, mb_rus, bnb_rus, spacy_model_rus)

    specialist, recommendation = recommend(p1, p2)

    data = {
        'timestamp': str(start_time),
        'content': content,
        'ratings': {
            'vote_tone': str(p1),
            'vote_proba': str(p2), 
            'neuro_tone': str(p3), 
            'neuro_proba': str(p4)
        },
        'spec': specialist, 
        'recommendation': recommendation
    }
    with open(cur_path_json, 'w') as write_file:
        json.dump(data, write_file)

    try:
        os.remove(cur_path_txt)
    except:
        pass

    reply = '\n\n'.join([f'Сообщение получено в {data["timestamp"]} (UTC)',
                       f'Текст: {data["content"]}',
                       f'Предсказанная тональность по версии голосующей системы: {data["ratings"]["vote_tone"]}',
                       f'Степень уверенности в версии голосующей системы: {data["ratings"]["vote_proba"]}',
                       f'Предсказанная тональность по версии нейросети: {data["ratings"]["neuro_tone"]}',
                       f'Степень уверенности в версии нейросети: {data["ratings"]["neuro_proba"]}',
                       f'Рекомендованный специалист: {data["spec"]}',
                       f'Пояснения: {data["recommendation"]}'])
    return reply


def recommend(predicted_score, proba):
    specialists = {'Новичок, стажер': 'Комментарий имеет позитивную окраску - можно подключить новичка/стажера', 
                   'Специалист по конфликтам/Опытный сотрудник': 'Комментарий резко негативный. Рекомендуется направить клиента к специалисту по конфликтам или опытного сотрудника',
                   'Любой свободный сотрудник': 'Настроение комментария нейтрально либо слабо смещено в положительную сторону - выбор сотрудника не имеет значения'}
    if predicted_score == "Негативный отзыв":
        return 'Специалист по конфликтам/Опытный сотрудник', specialists['Специалист по конфликтам/Опытный сотрудник']
    elif predicted_score == "Положительный отзыв":
        return 'Новичок, стажер', specialists['Новичок, стажер']
    else: 
        return 'Любой свободный сотрудник', specialists['Любой свободный сотрудник']
