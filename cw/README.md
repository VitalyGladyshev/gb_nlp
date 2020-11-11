# Курсовой проект по предмету "Введение в обработку естественного языка"
## Бот NLP Ответы MAIL.RU (@glvv_nlp_cw_bot)

Бот предоставляет доступ к базе вопросов-ответов сервиса Ответы Mail.ru в диалоговом режиме.

Бот выполняется на виртуальной машине в Яндекс Облаке (доля CPU 5%, 3Гб RAM), расчеты выполнены в Яндекс DataSphere (на S машине). Пока работает 24/7 :)
В бот добавлен Dialogflow и если нет признака вопроса, то запрос обрабатывается через него.

__Файл: NLP_otvety_prep.ipynb__ подготовка эмбедингов word2vec 300, fasttext 300, базы вопросов-ответов

__Файл: nlp_cw_bot.py__ исполняемый файл бота