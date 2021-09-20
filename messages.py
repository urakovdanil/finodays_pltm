say_hi = """Привет!
Я - бот команды ПЛТМ. Был разработан для участия в отборочном этапе хакатона Finodays. 

Моя задача - определение эмоциональной окраски входящих комментариев. Сейчас я делаю это двумя способами: при помощи голосующей системы и сверточной нейросети. 
После анализа комментариев я даю рекомендацию о том, с каким специалистом лучше соединить клиента. Рекомендации основываются на рейтинге голосующей системы. Оценку от нейросети я также предоставляю. Это нужно для того, чтобы в твоем распоряжении было несколько точек зрения. 

Для быстрой проверки функционала нажми на одну из кнопок ниже: я отправлю самому себе одно из 10 тестовых сообщений на выбранном языке. 
В ответ ты получишь результаты анализа приведенного комментария и соответствующие рекомендации. 

Помимо этого, можешь написать произвольный комментарий и в ответ получить рекомендации уже по нему!

Чтобы получить контакты разработчиков, нажми на соответствующую кнопку внизу. 

Мой исходный код - открытый. Репозиторий на GitHub: https://github.com/urakovdanil/finodays_pltm
         """

authors = """Разработчики: 
Даниил Ураков. @urakov_danil
Стажер аналитик данных. Ozon.
Предобработка данных для обучения моделей и нейросетей; разработка бота и интеграция обученных моделей; организация хранения статистики.

Иван Свиридов. @Khinkali_with_lamb
Дата-инженер. Иннотех, ВТБ.
Предобработка данных для обучения моделей и нейросетей; обучение и подбор моделей; разработка архитектуры формирования рекомендаций. 
"""

test_messages_rus = {
    1: '''Обман и мошенничество плюс дикий непрофессионализм сотрудников. Делали рефинансирование автокредита и ипотеки - 
обещали ставку 9% вместо 11,5 % при условии пользования картой Мир от поста банка и регулярном внесении платежей. 
Пользуемся картой постоянно вносили два месяца исправно платежи - в итоге ставка снова 11,5 % , списание средств с 
карты за просроченные платежи , испорченная кредитная история . С десятой попытки связались с горе «специалистом», 
который ничем помочь не может и лишь принял заявку к рассмотрению . Отвратительный банк , не связывайтесь с ними .
Лично мы готовим иск в суд.''', 
    2: '''Мне предложили новую кредитную карту.Карта называется Элемент 120, 
мой лимит на ней 100 тысяч.Условия как у любой другой кредитки, только беспроцентный период больше почти в 2 раза.
Там он 4 месяца. В общем, карта довольно удобная.''', 
    3: '''Банк неплохой. Качество обслуживания оставляет желать лучшего, но зато по ставкам  ощутимо превосходит остальные. 
Думаю брать у них иппотеку, надеюсь все получится.''', 
    4: '''Странный банк в смс предлагали кредит сначала одну сумму, потом другую, потом третью. Все это под 5.6 процентов.
Решили взять.Пришли. Стали оформлять. с 1,1 млн, которые предлагали в смс,
урезали до 300тыс. Непонятный развод на ровном месте! посмотрел отзывы и все стало понятно.''', 
    5: '''В клиентском центре Почта Банка на Ленинском пр.11 работает молодой,но очень знающий Дмитрий П. ,
который помог мне в получении наследства . Попала я к Дмитрию после двух неудачных попыток с другими работниками
в других отделениях .После первого общения у меня появилась уверенность , что все получится ..Так и вышло.''', 
    6: '''Никогда не было задолжности по выплате кредита. Но с банка начинают звонить за 2 недели до платежа.
А смс шлют раз 5-6 до срока платежа. Очень раздражает. Почитал отзывы, не у одного меня такая ситуация.
Больше в жизни не возьму тут ничего.''', 
    7: '''Уговаривают пенсионеров получать пенсию по их карте Мир. В итоге если пенсию приносили 8-го числа, то у них переводят только 15-го. Звонила в отделение, ничем не смогли помочь
Собираюсь от них уходить в другой банк.''', 
}

test_messages_eng = {
    1: '''Deception and fraud, plus the wild unprofessionalism of employees. We did refinancing of a car loan and a mortgage-we
promised a rate of 9 percent instead of 11.5 percent, provided that we use the Mir card from the bank's post and make regular payments.
We use the card constantly made payments regularly for two months - as a result, the rate is again 11.5 %, funds are debited from the
card for overdue payments , a damaged credit history. On the tenth attempt, we contacted the mountain ""specialist"",
who can not help in any way and only accepted the application for consideration.
Now we are preparing a lawsuit in court.''',
    2: '''The bank is not bad. The quality of service leaves much to be desired, but the bank significantly exceeds the others in terms of rates.
I'm thinking of taking a mortgage from them, I hope everything will work out.''',
    3: '''A young,but very knowledgeable Dmitry P. works in the client center of the Post Bank at 11 Leninsky Ave.,
who helped me in obtaining an inheritance . I got to Dmitry after two unsuccessful attempts with other employees
in other departments .After the first conversation, I had confidence that everything would work out ..And so it turned out.''',
    4: '''There has never been a debt on the payment of the loan. But the bank starts calling 2 weeks before the payment.
And SMS is sent 5-6 times before the payment deadline. Very annoying. I read the reviews, I'm not the only one with such a situation.
I won't take anything else here in my life.''', 
    5: '''I have Bank checking and savings account for more than 25 years. However, I finally decided to 
close the account and switch service to another institution. They used to have superb customer service.
But the quality of customer service is horrible nowadays. Also, they let the fraudulent activities happen
without my authorization.''', 
    6: '''Bank destroyed my credit score by 34 points claiming I filed for bankruptcy which is not true. 
I have somehow have managed to pay off credit cards despite being laid off for 5months in 2020. 
I was starting to pay this one down and they cancelled it and then claimed I’m bankrupt even tho that was
not true and I have never missed a payment.''', 
    7: '''I’ve only been with Bank for a short time but I have been pleased with the customer service.
I normally use the app or website to get service but I have on occasion called in to get assistance.
They are all very responsive and helpful.''', 
    8: '''The convenience that Bank offers is unparalleled to other banks. 
The saving interest rate has been competitive during such challenging times.
Although the rates have been dropping, they are still one the top providing the best prices.''', 
}
