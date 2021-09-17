import telebot
from telebot import types
import joblib

import config
import messages
import processing as proc


bot = telebot.TeleBot(config.BOT_TOKEN)

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
print('8-Ready...')

print('Считал модели. Готов к сообщениям')

@bot.message_handler(commands=['start'])
def welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton("Английский язык")
    item2 = types.KeyboardButton("Русский язык")

    markup.add(item1, item2)
    bot.send_message(message.chat.id,
                     messages.say_hi,
                     reply_markup=markup)


@bot.message_handler(content_types=['text'])
def process_message(message):
    if message.chat.type == 'private':
        if message.text not in ('Английский язык', 'Русский язык'):
            reply = proc.message_process_main(message)
            bot.send_message(message.chat.id, reply)
        elif message.text == 'Английский язык':
            reply = proc.message_process_eng(classifier_eng_vote, ln_eng, mb_eng, bnb_eng)
            bot.send_message(message.chat.id, reply)
        elif message.text == 'Русский язык':
            reply = proc.message_process_rus(classifier_rus_vote, ln_rus, mb_rus, bnb_rus)
            bot.send_message(message.chat.id, reply)


# RUN
bot.polling(none_stop=True)
