import os
import io
import re
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, flash, session
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import math

# --- Конфигурация ---
app = Flask(__name__)
# Секретный ключ для использования сессий (нужно для flash сообщений и хранения данных между запросами)
app.config['SECRET_KEY'] = os.urandom(24)
# Пагинация: количество элементов на странице
ITEMS_PER_PAGE = 50


# --- Вспомогательные функции ---

def preprocess_text(text):
    """
    Простая предварительная обработка текста:
    - Приведение к нижнему регистру
    - Удаление пунктуации и цифр (оставляем только слова)
    - Удаление слишком коротких слов (длиной 1-2 символа)
    """
    text = text.lower()
    # Оставляем только слова (буквы и, возможно, дефисы внутри слов)
    words = re.findall(r'\b[a-zа-яё-]+\b', text)
    # Фильтруем короткие слова
    words = [word for word in words if len(word) > 2]
    return " ".join(words) # Возвращаем строку для Vectorizer

def calculate_tf_idf(text_content):
    """
    Рассчитывает TF и IDF для слов в тексте.

    Возвращает:
        list: Список словарей вида {'word': слово, 'tf': частота, 'idf': значение idf}
              или None в случае ошибки.
        int: Общее количество уникальных слов до среза.
    """
    if not text_content.strip():
        return [], 0 # Возвращаем пустой список, если текст пустой

    processed_text = preprocess_text(text_content)
    words_list = processed_text.split() # Получаем список слов для Counter

    if not words_list:
        return [], 0 # Если после обработки слов не осталось

    # 1. Рассчитываем TF (Term Frequency) - сколько раз каждое слово встречается
    # Используем Counter для простого подсчета
    tf_counts = Counter(words_list)

    # 2. Рассчитываем IDF (Inverse Document Frequency) с помощью scikit-learn
    # TfidfVectorizer подходит для TF-IDF, но для получения чистого IDF удобнее использовать
    # его внутренний механизм. В контексте одного документа IDF будет одинаковым для всех слов.
    # Используем stop_words='english' или 'russian' если известен язык,
    # но для универсальности пока не будем их жестко задавать,
    # т.к. предобработка уже сделала часть работы.
    # Важно: TfidfVectorizer внутри себя тоже выполняет токенизацию и предобработку,
    # поэтому важно, чтобы она была согласована с нашей ручной предобработкой (или полагаться только на Vectorizer).
    # Здесь для демонстрации мы используем его IDF компонент.

    try:
        # Используем TfidfVectorizer для расчета IDF.
        # smooth_idf=True и use_idf=True включены по умолчанию.
        # Нормализация ('norm') не влияет на сам IDF.
        # Мы передаем текст как список из одного документа.
        vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, token_pattern=r'\b[a-zа-яё-]+\b', min_df=1)

        # Обучаем векторизатор на нашем тексте (одном документе)
        tfidf_matrix = vectorizer.fit_transform([processed_text])

        # Получаем словарь {слово: индекс}
        feature_names = vectorizer.get_feature_names_out()
        # Получаем значения IDF для каждого слова в словаре
        idf_values = vectorizer.idf_

        # Создаем словарь {слово: idf}
        idf_dict = dict(zip(feature_names, idf_values))

        # 3. Собираем результаты
        results = []
        for word in feature_names: # Итерируем по словам, найденным векторизатором
            if word in tf_counts: # Убеждаемся, что слово есть в наших TF подсчетах
                 results.append({
                     'word': word,
                     'tf': tf_counts[word],
                     'idf': idf_dict.get(word, 0.0) # Используем get для безопасности
                 })

        # 4. Сортируем результаты по убыванию IDF
        # ВАЖНО: Как отмечено, для одного документа IDF будет одинаковым (1.0) для всех слов.
        # Поэтому сортировка по IDF не даст уникального ранжирования.
        # Можно добавить вторичную сортировку, например, по TF (убывание) или по слову (алфавит).
        # results.sort(key=lambda item: (-item['idf'], -item['tf'], item['word']))
        results.sort(key=lambda item: item['idf'], reverse=True)

        total_unique_words = len(results)

        # Возвращаем все результаты и общее количество
        return results, total_unique_words

    except Exception as e:
        print(f"Ошибка при расчете TF-IDF: {e}")
        return None, 0


# --- Маршруты Flask ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Главная страница: обработка GET (показ формы) и POST (обработка файла).
    """
    if request.method == 'POST':
        # Проверка наличия файла в запросе
        if 'textFile' not in request.files:
            flash('Файл не был отправлен.', 'error')
            return redirect(request.url)

        file = request.files['textFile']

        # Проверка имени файла
        if file.filename == '':
            flash('Файл не выбран.', 'error')
            return redirect(request.url)

        # Проверка типа файла (простое расширение)
        if file and file.filename.lower().endswith('.txt'):
            try:
                # Читаем содержимое файла безопасно
                # Используем io.BytesIO и декодируем в UTF-8 (самая частая кодировка)
                # Добавляем обработку ошибок декодирования
                file_content_bytes = file.read()
                try:
                    file_content = file_content_bytes.decode('utf-8')
                except UnicodeDecodeError:
                     try:
                         # Попытка с другой распространенной кодировкой
                         file_content = file_content_bytes.decode('cp1251')
                     except UnicodeDecodeError:
                         flash('Не удалось прочитать файл. Убедитесь, что он в кодировке UTF-8 или CP1251.', 'error')
                         return redirect(request.url)

                # Рассчитываем TF-IDF
                all_results, total_words = calculate_tf_idf(file_content)

                if all_results is None:
                    flash('Произошла ошибка при обработке файла.', 'error')
                    return redirect(request.url)

                # Сохраняем все результаты в сессию для пагинации
                session['all_results'] = all_results
                session['total_words'] = total_words
                session['filename'] = file.filename

                # Перенаправляем на GET запрос первой страницы результатов
                return redirect(url_for('index', page=1))

            except Exception as e:
                flash(f'Произошла внутренняя ошибка: {e}', 'error')
                return redirect(request.url)
        else:
            flash('Пожалуйста, загрузите файл с расширением .txt', 'error')
            return redirect(request.url)

    # Обработка GET запроса (отображение формы и результатов, если они есть)
    page = request.args.get('page', 1, type=int) # Получаем номер страницы из URL
    results_to_display = None
    pagination_data = None
    filename = session.get('filename') # Получаем имя файла из сессии

    if 'all_results' in session:
        all_results = session['all_results']
        total_words = session.get('total_words', 0)

        # Логика пагинации
        start_index = (page - 1) * ITEMS_PER_PAGE
        end_index = start_index + ITEMS_PER_PAGE
        results_to_display = all_results[start_index:end_index]

        total_pages = math.ceil(total_words / ITEMS_PER_PAGE)

        # Формируем данные для пагинации в шаблоне
        pagination_data = {
            'page': page,
            'per_page': ITEMS_PER_PAGE,
            'total': total_words,
            'total_pages': total_pages,
            'has_prev': page > 1,
            'has_next': page < total_pages,
            'prev_num': page - 1 if page > 1 else None,
            'next_num': page + 1 if page < total_pages else None,
            # Простая генерация номеров страниц для отображения (можно улучшить)
            'iter_pages': lambda: range(1, total_pages + 1)
        }

        # Показываем только топ N если пагинация не нужна была бы
        # results_to_display = all_results[:ITEMS_PER_PAGE]

        # Если на текущей странице нет результатов (например, запрошена страница > max),
        # но результаты вообще есть, перенаправим на первую
        if not results_to_display and page > 1 and all_results:
             return redirect(url_for('index', page=1))


    # Получаем flash сообщения
    error_messages = session.pop('_flashes', []) # Используем стандартный механизм flash

    # Отображаем шаблон
    return render_template(
        'index.html',
        results=results_to_display,
        pagination=pagination_data,
        total_words=session.get('total_words'),
        filename=filename,
        error=next((msg[1] for msg in error_messages if msg[0] == 'error'), None) # Передаем только последнее сообщение об ошибке
        # Можно передать все сообщения, если нужно
    )

# --- Запуск приложения ---
if __name__ == '__main__':
    # debug=True полезен для разработки, но его следует отключить в продакшене
    app.run(debug=True)