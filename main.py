import telebot
from telebot import types
from telebot import apihelper
import joblib
from langdetect import detect
import spacy
from time import time, sleep
import datetime
import json
import os

import config
import messages
import processing as proc


begin = time()
print('Считываю модели...', end=' ')
classifier_eng_vote = joblib.load("sentiment_classifier_vot_eng.pkl")
print('1-Ready...', end=' ')
ln_eng = joblib.load("lr_eng.pkl")
print('2-Ready...', end=' ')
mb_eng = joblib.load("mnb_eng.pkl")
print('3-Ready...', end=' ')
bnb_eng = joblib.load("bnb_eng.pkl")
print('4-Ready...', end=' ')

classifier_rus_vote = joblib.load("sentiment_classifier_vot_rus.pkl")
print('5-Ready...', end=' ')
ln_rus = joblib.load("lr_rus.pkl")
print('6-Ready...', end=' ')
mb_rus = joblib.load("mnb_rus.pkl")
print('7-Ready...', end=' ')
bnb_rus = joblib.load("bnb_rus.pkl")
print('8-Ready...', end=' ')

spacy_model = spacy.load("model_artifacts")
print('9-Ready...', end=' ')
spacy_model_rus = spacy.load("model_artifacts_rus")
print('10-Ready...')

print('Считал модели. Готов к сообщениям')

bot = telebot.TeleBot(config.BOT_TOKEN)

end = time()
mess = 'Бот запущен\n' + 'Время запуска: ' + str((end - begin) / 60)
bot.send_message(649539206, mess)
bot.send_message(860903600, mess)

@bot.message_handler(commands=['start'])
def welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton("Английский язык")
    item2 = types.KeyboardButton("Русский язык")
    item3 = types.KeyboardButton("Контакты авторов")

    markup.add(item1, item2, item3)
    bot.send_message(message.chat.id,
                     messages.say_hi,
                     reply_markup=markup)


@bot.message_handler(content_types=['text'])
def process_message(message):
    if message.chat.type == 'private':
        if message.text not in ('Английский язык', 'Русский язык', "Контакты авторов"):
            start_time = datetime.datetime.today().replace(microsecond=0)
            cur_path_txt = str(round(100 * time())) + '.txt'
            cur_path_json = str(round(100 * time())) + '.json'
            with open(cur_path_txt, 'w', encoding='utf-8') as cur_message:
                cur_message.write(message.text)

            with open(cur_path_txt, 'r', encoding='utf-8') as f:
                content = '\n'.join(f.readlines())
            bot.send_message(message.chat.id, 'Принято. Обрабатываю...')
            if detect(content) == 'ru':
                p1, p2, p3, p4 = proc.predict_rus(content, classifier_rus_vote, ln_rus, mb_rus, bnb_rus, spacy_model_rus)

                specialist, recommendation = proc.recommend(p1, p2)

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

                reply = '\n'.join([f'Сообщение получено в {data["timestamp"]} (UTC)\n',
                                   f'Текст: {data["content"]}\n',
                                   f'Предсказанная тональность по версии голосующей системы: {data["ratings"]["vote_tone"]}',
                                   f'Степень уверенности в версии голосующей системы: {data["ratings"]["vote_proba"]}\n',
                                   f'Предсказанная тональность по версии нейросети: {data["ratings"]["neuro_tone"]}',
                                   f'Степень уверенности в версии нейросети: {data["ratings"]["neuro_proba"]}\n',
                                   f'Рекомендованный специалист: {data["spec"]}',
                                   f'Пояснения: {data["recommendation"]}'])

                bot.send_message(message.chat.id, reply)
            else:
                p1, p2, p3, p4 = proc.predict(content, classifier_eng_vote, ln_eng, mb_eng, bnb_eng, spacy_model)

                specialist, recommendation = proc.recommend(p1, p2)

                data = {
                    'timestamp': str(start_time),
                    'content': message.text,
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

                reply = '\n\n'.join([f'Сообщение получено в {data["timestamp"]} (UTC)\n',
                                   f'Текст: {data["content"]}\n',
                                   f'Предсказанная тональность по версии голосующей системы: {data["ratings"]["vote_tone"]}',
                                   f'Степень уверенности в версии голосующей системы: {data["ratings"]["vote_proba"]}\n',
                                   f'Предсказанная тональность по версии нейросети: {data["ratings"]["neuro_tone"]}',
                                   f'Степень уверенности в версии нейросети: {data["ratings"]["neuro_proba"]}\n',
                                   f'Рекомендованный специалист: {data["spec"]}',
                                   f'Пояснения: {data["recommendation"]}'])
                bot.send_message(message.chat.id, reply)   
        elif message.text == 'Английский язык':
            bot.send_message(message.chat.id, 'Принято. Обрабатываю...')
            reply = proc.message_process_eng(classifier_eng_vote, ln_eng, mb_eng, bnb_eng, spacy_model)
            bot.send_message(message.chat.id, reply)
        elif message.text == 'Русский язык':
            bot.send_message(message.chat.id, 'Принято. Обрабатываю...')
            reply = proc.message_process_rus(classifier_rus_vote, ln_rus, mb_rus, bnb_rus, spacy_model_rus)
            bot.send_message(message.chat.id, reply)
        elif message.text == 'Контакты авторов':
            bot.send_message(message.chat.id, messages.authors)


# RUN

fell_down = False
while True:
    try:
        bot.polling(none_stop=True)
        if fell_down == True:
            bot.send_message(649539206, 'Падал, восстановил работу')
            fell_down = False

    except Exception as e:
        print(e)
        fell_down = True
        sleep(15)
