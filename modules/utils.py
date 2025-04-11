import streamlit as st
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_custom_css():
    """
    Применяет пользовательские CSS-стили из файла static/styles.css к приложению Streamlit.
    """
    # Определяем путь к файлу CSS относительно текущего файла
    css_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'styles.css')

    # Проверяем наличие файла CSS
    if os.path.exists(css_path):
        try:
            with open(css_path, 'r', encoding='utf-8') as f:
                css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
            logger.info("Custom CSS applied successfully from %s", css_path)
        except Exception as e:
            logger.error("Error reading or applying CSS file %s: %s", css_path, e)
            st.error(f"Не удалось загрузить стили оформления: {e}")
    else:
        # Если файл не найден, выводим предупреждение
        logger.warning("CSS file not found at %s. Using default Streamlit styles.", css_path)
        st.warning("Файл стилей styles.css не найден. Используются стандартные стили.")

def format_number(number, precision=2):
    """
    Форматирует число с разделителями тысяч и указанной точностью
    """
    if isinstance(number, (int, float)):
        if number == int(number):
            return f"{int(number):,}".replace(",", " ")
        else:
            return f"{number:,.{precision}f}".replace(",", " ")
    return str(number)

def get_download_link(df, filename, text):
    """
    Создает ссылку для скачивания DataFrame в формате Excel
    """
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Данные', index=False)
    
    # Автоподбор ширины столбцов
    workbook = writer.book
    worksheet = writer.sheets['Данные']
    
    for i, col in enumerate(df.columns):
        column_width = max(df[col].astype(str).map(len).max(), len(col)) + 2
        worksheet.set_column(i, i, column_width)
    
    writer.close()
    
    b64 = base64.b64encode(output.getvalue()).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{text}</a>'

def create_heatmap(pivot_data, title):
    """
    Создает тепловую карту из сводной таблицы
    """
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale="Viridis",
        colorbar=dict(title="Значение"),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        height=500,
        yaxis_title="Строки",
        xaxis_title="Столбцы"
    )
    
    return fig

def create_correlation_matrix(df, columns, title="Корреляционная матрица"):
    """
    Создает корреляционную матрицу для выбранных столбцов
    """
    # Выбираем только числовые столбцы из списка
    numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_columns:
        return None
    
    # Вычисляем корреляцию
    corr_matrix = df[numeric_columns].corr()
    
    # Создаем тепловую карту
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu_r",
        zmid=0,  # Центр шкалы в 0
        colorbar=dict(title="Корреляция"),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        height=600,
        width=700,
        yaxis_title="",
        xaxis_title=""
    )
    
    return fig

def plot_time_series_decomposition(series, title="Декомпозиция временного ряда"):
    """
    Создает график декомпозиции временного ряда на тренд, сезонность и остаток
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Проверяем, что временной ряд достаточной длины для декомпозиции
    if len(series) < 4:
        return None
    
    # Выполняем декомпозицию
    try:
        # Определяем период сезонности
        freq = series.index.freq
        period = 12  # По умолчанию для месячных данных
        
        if freq == 'D':
            period = 7  # Для дневных данных (неделя)
        elif freq == 'W':
            period = 52  # Для недельных данных (год)
        elif freq == 'Q':
            period = 4  # Для квартальных данных
        
        # Выполняем декомпозицию
        decomposition = seasonal_decompose(
            series, 
            model='additive', 
            period=min(period, len(series) // 2)  # Не более половины длины ряда
        )
        
        # Создаем фигуру с подграфиками
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))
        
        # Исходный ряд
        decomposition.observed.plot(ax=axes[0])
        axes[0].set_title('Исходный ряд')
        axes[0].set_xlabel('')
        
        # Тренд
        decomposition.trend.plot(ax=axes[1])
        axes[1].set_title('Тренд')
        axes[1].set_xlabel('')
        
        # Сезонность
        decomposition.seasonal.plot(ax=axes[2])
        axes[2].set_title('Сезонность')
        axes[2].set_xlabel('')
        
        # Остаток
        decomposition.resid.plot(ax=axes[3])
        axes[3].set_title('Остаток')
        
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        st.error(f"Ошибка при декомпозиции временного ряда: {str(e)}")
        return None

def calculate_min_forecast_points(freq, horizon):
    """
    Вычисляет минимальное количество точек данных для прогнозирования
    на основе частоты и горизонта прогнозирования
    """
    # Минимальное количество точек для разных частот
    min_points = {
        'D': max(14, horizon),       # Для дневных данных
        'W': max(8, horizon),        # Для недельных данных
        'M': max(6, horizon),        # Для месячных данных
        'Q': max(4, horizon),        # Для квартальных данных
        'Y': max(3, horizon)         # Для годовых данных
    }
    
    # Возвращаем минимальное количество точек или 2*horizon, если частота не указана
    return min_points.get(freq, max(5, 2 * horizon))

def detect_outliers(series, method='iqr', threshold=1.5):
    """
    Обнаруживает выбросы во временном ряду
    
    Args:
        series: pandas Series с данными
        method: метод обнаружения выбросов ('iqr' или 'zscore')
        threshold: порог для определения выбросов
    
    Returns:
        pandas Series с булевыми значениями (True для выбросов)
    """
    if method == 'iqr':
        # Метод межквартильного размаха
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == 'zscore':
        # Метод z-оценки
        mean = series.mean()
        std = series.std()
        
        if std == 0:  # Предотвращаем деление на ноль
            return pd.Series(False, index=series.index)
        
        z_scores = (series - mean) / std
        
        return abs(z_scores) > threshold
    
    else:
        # Если метод не распознан, возвращаем все False
        return pd.Series(False, index=series.index)

def format_streamlit_dataframe(df, height=None):
    """
    Форматирует DataFrame для отображения в Streamlit
    
    Args:
        df: pandas DataFrame
        height: высота таблицы в пикселях (None - автоматически)
        
    Returns:
        стилизованный DataFrame
    """
    # Применяем базовый стиль к таблице
    styled_df = df.style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#E3F2FD'), 
                                     ('color', '#000000'), 
                                     ('font-weight', 'bold'),
                                     ('border', '1px solid #B0BEC5')]},
        {'selector': 'td', 'props': [('border', '1px solid #E0E0E0')]},
        {'selector': 'tr:hover', 'props': [('background-color', '#F5F5F5')]},
        {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#FAFAFA')]}
    ])
    
    # Форматируем числовые столбцы
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    for col in numeric_cols:
        if 'процент' in col.lower() or 'доля' in col.lower() or '%' in col:
            # Форматируем как процент с двумя десятичными знаками
            styled_df = styled_df.format({col: lambda x: f"{x:.2f}%"})
        elif 'цена' in col.lower() or 'стоимость' in col.lower() or 'руб' in col.lower():
            # Используем функцию format_number для форматирования чисел с разделителями тысяч
            styled_df = styled_df.format({col: lambda x: format_number(x, precision=2)})
        elif df[col].dtype == 'int64':
            # Целые числа форматируем с разделителями тысяч
            styled_df = styled_df.format({col: lambda x: format_number(x, precision=0)})
        else:
            # Прочие числа форматируем с двумя десятичными знаками
            styled_df = styled_df.format({col: lambda x: format_number(x, precision=2)})
    
    # Опционально можно выделить цветом ячейки на основе значений
    # Например, выделить отрицательные значения красным
    def highlight_negative(val):
        if isinstance(val, (int, float)) and val < 0:
            return 'color: red'
        return ''
    
    styled_df = styled_df.applymap(highlight_negative)
    
    return styled_df

def show_error_message(error, error_type="Ошибка", show_traceback=False):
    """
    Отображает сообщение об ошибке в приятном для пользователя формате
    
    Args:
        error: объект исключения или текст ошибки
        error_type: тип ошибки (например, "Ошибка загрузки данных")
        show_traceback: показывать ли трассировку стека
    """
    import traceback
    
    # Главное сообщение об ошибке
    error_html = f"""
    <div style="background-color: #FFEBEE; padding: 15px; border-radius: 5px; margin: 10px 0;">
        <h3 style="color: #B71C1C; margin-top: 0;">{error_type}</h3>
        <p style="margin-bottom: 5px; font-size: 16px;">{str(error)}</p>
    """
    
    # Добавляем советы по решению типичных проблем
    if "encoding" in str(error).lower() or "encod" in str(error).lower():
        error_html += """
        <div style="background-color: #FFF3E0; padding: 10px; border-radius: 3px; margin-top: 10px;">
            <p style="margin: 0; font-weight: bold;">Возможные решения:</p>
            <ul style="margin-top: 5px;">
                <li>Проверьте кодировку файла (UTF-8, Windows-1251)</li>
                <li>Исключите из файла нестандартные символы</li>
                <li>Убедитесь, что файл не поврежден</li>
            </ul>
        </div>
        """
    elif "memory" in str(error).lower() or "memory" in str(error).lower():
        error_html += """
        <div style="background-color: #FFF3E0; padding: 10px; border-radius: 3px; margin-top: 10px;">
            <p style="margin: 0; font-weight: bold;">Возможные решения:</p>
            <ul style="margin-top: 5px;">
                <li>Попробуйте загрузить файл меньшего размера</li>
                <li>Увеличьте ограничение памяти в настройках приложения</li>
                <li>Разделите большой файл на несколько меньших</li>
            </ul>
        </div>
        """
    elif "файл" in str(error).lower() or "file" in str(error).lower():
        error_html += """
        <div style="background-color: #FFF3E0; padding: 10px; border-radius: 3px; margin-top: 10px;">
            <p style="margin: 0; font-weight: bold;">Возможные решения:</p>
            <ul style="margin-top: 5px;">
                <li>Проверьте формат и структуру файла</li>
                <li>Убедитесь, что файл содержит все необходимые колонки</li>
                <li>Попробуйте открыть файл в Excel и сохранить в формате CSV</li>
            </ul>
        </div>
        """
    
    # Добавляем трассировку стека, если запрошено
    if show_traceback:
        stack_trace = traceback.format_exc()
        error_html += f"""
        <details>
            <summary style="cursor: pointer; color: #616161; margin-top: 10px;">Показать техническую информацию</summary>
            <pre style="background-color: #F5F5F5; padding: 10px; border-radius: 3px; margin-top: 5px; 
                      white-space: pre-wrap; font-size: 12px; color: #212121;">{stack_trace}</pre>
        </details>
        """
    
    error_html += "</div>"
    
    # Отображаем HTML
    st.markdown(error_html, unsafe_allow_html=True)

def show_loading_spinner(message="Загрузка данных...", key=None):
    """
    Отображает анимированный индикатор загрузки с сообщением
    
    Args:
        message: текст сообщения
        key: уникальный ключ для виджета (если нужно несколько индикаторов)
    
    Returns:
        placeholder: объект-заполнитель для сообщения
    """
    spinner_html = f"""
    <div style="display: flex; align-items: center; margin: 10px 0;">
        <div class="loading-spinner"></div>
        <span style="margin-left: 10px; color: #424242;">{message}</span>
    </div>
    
    <style>
    .loading-spinner {{
        border: 4px solid #f3f3f3;
        border-radius: 50%;
        border-top: 4px solid #1E88E5;
        width: 24px;
        height: 24px;
        animation: spinner-rotation 1s linear infinite;
    }}
    
    @keyframes spinner-rotation {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    </style>
    """
    
    placeholder = st.empty() if key is None else st.empty().key(key)
    placeholder.markdown(spinner_html, unsafe_allow_html=True)
    return placeholder

def show_export_success(filename, filesize, duration):
    """
    Отображает сообщение об успешном экспорте данных
    
    Args:
        filename: имя файла
        filesize: размер файла в байтах
        duration: продолжительность экспорта в секундах
    """
    # Преобразуем размер в читаемый формат
    size_str = ""
    if filesize < 1024:
        size_str = f"{filesize} байт"
    elif filesize < 1024 * 1024:
        size_str = f"{filesize / 1024:.1f} КБ"
    else:
        size_str = f"{filesize / (1024 * 1024):.1f} МБ"
    
    success_html = f"""
    <div style="background-color: #E8F5E9; padding: 15px; border-radius: 5px; margin: 10px 0;">
        <h3 style="color: #2E7D32; margin-top: 0; margin-bottom: 10px;">Экспорт данных завершен</h3>
        <p style="margin: 0 0 5px 0;"><strong>Файл:</strong> {filename}</p>
        <p style="margin: 0 0 5px 0;"><strong>Размер:</strong> {size_str}</p>
        <p style="margin: 0 0 5px 0;"><strong>Время выполнения:</strong> {duration:.2f} сек.</p>
    </div>
    """
    
    st.markdown(success_html, unsafe_allow_html=True)

def create_styled_dataframe(df, height=None, precision=2, highlight_cols=None, highlight_threshold=None):
    """
    Создает стилизованный DataFrame для отображения в Streamlit
    
    Args:
        df: pandas DataFrame для отображения
        height: высота таблицы в пикселях (None - автоматически)
        precision: точность для числовых значений
        highlight_cols: список колонок для подсветки значений
        highlight_threshold: пороговое значение для подсветки
        
    Returns:
        styled_df: стилизованный DataFrame с форматированием
    """
    # Создаем копию DataFrame для форматирования
    formatted_df = df.copy()
    
    # Функция для форматирования числовых значений
    def format_value(x, precision=precision):
        if pd.isna(x):
            return ""
        elif isinstance(x, (int, np.integer)):
            return f"{x:,}".replace(",", " ")
        elif isinstance(x, (float, np.floating)):
            if x == int(x):  # Если число без десятичной части
                return f"{int(x):,}".replace(",", " ")
            else:
                return f"{x:,.{precision}f}".replace(",", " ")
        else:
            return str(x)
    
    # Применяем форматирование к числовым колонкам
    for col in formatted_df.select_dtypes(include=np.number).columns:
        formatted_df[col] = formatted_df[col].apply(format_value)
    
    # Создаем объект стиля
    styled_df = formatted_df.style
    
    # Добавляем базовое форматирование таблицы
    styled_df = styled_df.set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#E3F2FD'), 
            ('color', '#0D47A1'), 
            ('font-weight', 'bold'),
            ('border', '1px solid #B0BEC5'),
            ('padding', '8px'),
            ('text-align', 'left'),
            ('white-space', 'nowrap'),
            ('overflow', 'hidden'),
            ('text-overflow', 'ellipsis'),
            ('max-width', '200px')
        ]},
        {'selector': 'td', 'props': [
            ('border', '1px solid #E0E0E0'),
            ('padding', '8px'),
            ('max-width', '200px'),
            ('overflow', 'hidden'),
            ('text-overflow', 'ellipsis'),
            ('white-space', 'nowrap')
        ]},
        {'selector': 'tr:hover', 'props': [
            ('background-color', '#E3F2FD')
        ]},
        {'selector': 'tr:nth-child(even)', 'props': [
            ('background-color', '#F5F5F5')
        ]},
        {'selector': 'caption', 'props': [
            ('caption-side', 'bottom'), 
            ('font-style', 'italic'),
            ('color', '#616161'),
            ('padding', '8px'),
            ('text-align', 'left')
        ]},
        # Стиль для всей таблицы
        {'selector': '', 'props': [
            ('border-collapse', 'collapse'),
            ('font-family', 'Arial, sans-serif'),
            ('width', '100%')
        ]}
    ])
    
    # Если указаны колонки для подсветки и порог
    if highlight_cols and highlight_threshold is not None:
        # Преобразуем в список, если передана одна колонка
        if not isinstance(highlight_cols, list):
            highlight_cols = [highlight_cols]
            
        # Подсветка значений выше порога
        for col in highlight_cols:
            if col in df.columns:
                styled_df = styled_df.apply(
                    lambda x: ['background-color: #FFEB9C' if (not pd.isna(y) and float(str(y).replace(' ', '').replace(',', '.')) > highlight_threshold) else '' 
                            for y in x], 
                    subset=[col]
                )
    
    # Добавляем подпись с информацией о размере таблицы
    caption = f"Всего строк: {len(df)}, колонок: {len(df.columns)}"
    styled_df = styled_df.set_caption(caption)
    
    return styled_df

def show_user_guide():
    """
    Отображает руководство пользователя по использованию приложения в формате Markdown
    """
    st.markdown("""
    ## Руководство пользователя

    Это приложение предназначено для анализа данных о материалах и их ценах, 
    а также для подготовки к прогнозированию цен с помощью различных методов машинного обучения.

    ### Основные разделы

    #### 1. Загрузка данных
    *   В этом разделе вы можете загрузить CSV-файл с данными о материалах. 
    *   **Обязательные колонки:** `Материал`, `ДатаСоздан`, `Цена нетто`, `за`, `ЕЦЗ`, `Влт`, `Курс`, `З-д`, `ДокумЗакуп`, `ГрЗ`, `ГруппаМтр`.
    *   Доступны настройки параметров импорта: кодировка (UTF-8, Windows-1251), разделитель столбцов (например, `;` или `,`), десятичный разделитель (`,` или `.`).
    *   **Пример:** Загрузите файл `purchases.csv` с кодировкой UTF-8 и разделителем `;`.

    #### 2. Общий анализ
    *   Отображает общую статистику по загруженным и обработанным данным:
        *   Общее количество записей.
        *   Количество уникальных материалов.
        *   Временной диапазон данных (самая ранняя и поздняя дата).
        *   Распределение записей по годам.
        *   Статистика по используемым валютам (`Влт`).
        *   Базовая статистика по ценам (`Цена нетто`).
    *   **Пример:** Показывает, что загружено 182067 записей с 01.01.2022 по 31.12.2024, большинство записей в RUB.

    #### 3. Анализ уникальности материалов
    *   Анализирует, сколько записей приходится на каждый уникальный материал.
    *   Позволяет выявить:
        *   Материалы с наибольшим количеством записей (самые часто закупаемые).
        *   Материалы с одной записью (редкие или разовые закупки).
        *   Общее распределение: много ли материалов с малым числом записей и наоборот.
    *   **Пример:** Материал "Болт М12" имеет 500 записей, а "Специальный реагент X" - только 1.

    #### 4. Временной анализ
    *   Анализирует динамику цен и закупок во времени.
    *   Для выбранного материала можно увидеть:
        *   График изменения цены (`Цена нетто`) со временем.
        *   Сезонность (если присутствует).
        *   Статистику по временным интервалам между закупками (как часто закупается).
    *   **Пример:** Показывает, что цена на "Масло моторное" имеет тенденцию к росту к концу года.

    #### 5. Анализ волатильности
    *   Анализирует изменчивость цен материалов с помощью коэффициента вариации (CV).
    *   Позволяет определить:
        *   Материалы с наиболее стабильными ценами (низкий CV).
        *   Материалы с наиболее нестабильными ценами (высокий CV).
        *   Общее распределение волатильности по всем материалам.
    *   **Пример:** "Гвозди" имеют CV=5% (стабильная цена), а "Медь листовая" - CV=45% (высокая волатильность).

    #### 6. Стабильные материалы
    *   Выделяет материалы, цена которых почти не меняется.
    *   По умолчанию ищет материалы, где 80% или более записей имеют одинаковую цену.
    *   **Пример:** Материал "Бумага А4" имеет одну и ту же цену в 90% закупок.

    #### 7. Неактивные материалы
    *   Находит материалы, по которым давно не было закупок.
    *   Можно настроить порог неактивности (например, 365 дней).
    *   **Пример:** Материал "Картридж HP 123" не закупался последние 500 дней.

    #### 8. Сегментация для прогнозирования
    *   Разделяет материалы на группы (сегменты) на основе их характеристик для выбора наилучшего метода прогнозирования цен.
    *   **Параметры сегментации:**
        *   `Минимальное количество точек данных:` Сколько записей должно быть у материала для сложного прогноза (по умолчанию 24).
        *   `Максимальный коэффициент вариации (%):` Насколько цена может колебаться для ML-прогноза (по умолчанию 30%).
        *   `Минимальное количество дней активности:` Общая продолжительность истории закупок (по умолчанию 365 дней).
    *   **Сегменты:**
        *   `ML-прогнозирование:` Достаточно данных, умеренная волатильность, достаточная история. **Подходят для:** Сложных моделей (ARIMA, Prophet, LSTM). **Пример:** "Сталь листовая" - 100 записей за 2 года, CV=15%.
        *   `Наивные методы:` Достаточно данных, но слишком высокая волатильность или короткая история. **Подходят для:** Простых моделей (среднее, последняя цена, сезонное среднее). **Пример:** "Электронный компонент Y" - 50 записей за 1 год, CV=50%.
        *   `Постоянная цена:` Цена почти не меняется (очень низкий CV). **Подходят для:** Использования текущей цены как прогноза. **Пример:** "Канцелярский скотч" - 200 записей, CV=1%.
        *   `Неактивные:` Закупок не было дольше установленного порога. **Прогноз невозможен** без новых данных. **Пример:** "Лампа накаливания 60Вт" - последняя закупка 2 года назад.
        *   `Недостаточно истории:` Слишком мало записей для надежного прогноза. **Прогноз не рекомендуется**. **Пример:** "Экспериментальный сплав Z" - 3 записи за 1 месяц.
        *   `Высокая волатильность:` Цена изменяется слишком сильно и непредсказуемо. **Прогноз затруднен**, требует экспертной оценки или других подходов. **Пример:** "Акции компании TechCorp" - 25 записей, CV=70%.

    #### 9. Анализ безопасности
    *   Выявляет подозрительные паттерны в закупках, которые могут указывать на риски (мошенничество, ошибки).
    *   **Анализируемые индикаторы:**
        *   Высокая волатильность цен (резкие скачки).
        *   Признаки дробления закупок (много мелких закупок ниже порогов).
        *   Повышенная активность в конце отчетных периодов (месяц, квартал).
        *   Подозрительно округленные цены (например, 10000.00 вместо 9985.50).
        *   Аномальная стабильность для рыночных товаров.
    *   Материалы ранжируются по уровню риска (Низкий, Средний, Высокий).
    *   Для материалов с высоким риском доступен **детальный анализ** с графиками и таблицами аномалий.
    *   **Пример:** Обнаружено, что закупки "Кабель медный" часто происходят мелкими партиями по 49900 руб (при пороге в 50000) в последние дни квартала.

    #### 10. Экспорт данных
    *   Позволяет выгрузить результаты анализа и сегментации для использования в других системах или для отчетности.
    *   **Опции экспорта:**
        *   `Экспорт по сегментам:` Скачать данные для одного выбранного сегмента (например, только 'ML-прогнозирование') в CSV или Excel.
        *   `Массовый экспорт:` Скачать данные для всех сегментов одним ZIP-архивом, где каждый сегмент - отдельный файл.
        *   `Настраиваемый экспорт:` (Может быть добавлено в будущем) - Экспорт с дополнительными фильтрами.
    *   **Пример:** Экспортировать список материалов из сегмента 'ML-прогнозирование' в Excel для передачи в систему прогнозирования.

    ### Советы по использованию

    1.  **Качество данных:** Успех анализа сильно зависит от качества исходных данных. Убедитесь, что файл содержит корректные данные, особенно `Материал`, `ДатаСоздан` и `Цена нетто`.
    2.  **Параметры сегментации:** Подбирайте параметры сегментации в разделе 8 в соответствии с вашими целями прогнозирования и особенностями данных. Экспериментируйте с порогами.
    3.  **Большие объемы:** При работе с очень большими файлами (сотни тысяч строк) анализ и сегментация могут занять время. Анализ безопасности для большого числа материалов с высоким риском также может быть долгим. Используйте фильтры или выбирайте ограниченное количество материалов для детального анализа.
    4.  **Интерпретация:** Не все материалы в сегменте 'Высокая волатильность' или с высоким риском безопасности обязательно являются проблемными. Это лишь индикаторы, требующие дополнительного изучения.
    5.  **Анализ безопасности:** Используйте детальный анализ для материалов с высоким риском, чтобы понять конкретные причины аномалий (скачки цен, дробление и т.д.).

    """, unsafe_allow_html=False) # Используем Markdown, HTML не разрешен

def show_performance_info():
    """
    Отображает информацию о производительности приложения
    """
    try:
        import psutil
        import gc
        
        # Запускаем сборку мусора
        gc.collect()
        
        # Получаем информацию о текущем процессе
        process = psutil.Process()
        
        # Получаем информацию о памяти
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024  # в МБ
        
        # Получаем информацию о CPU
        cpu_percent = process.cpu_percent(interval=0.1)
        
        # Получаем информацию о системе
        system_memory = psutil.virtual_memory()
        system_memory_usage_percent = system_memory.percent
        
        # Создаем HTML для вывода информации
        performance_html = f"""
        <div style="background-color: #E8F5E9; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 0.8em;">
            <h4 style="color: #2E7D32; margin-top: 0; margin-bottom: 5px;">Информация о производительности</h4>
            <p style="margin: 2px 0;"><strong>Использование памяти:</strong> {memory_usage_mb:.1f} МБ ({(memory_usage_mb/system_memory.total*100):.1f}% от доступной)</p>
            <p style="margin: 2px 0;"><strong>Загрузка CPU:</strong> {cpu_percent:.1f}%</p>
            <p style="margin: 2px 0;"><strong>Загрузка памяти системы:</strong> {system_memory_usage_percent:.1f}%</p>
            <p style="margin: 2px 0;"><strong>Доступно памяти:</strong> {(system_memory.available/1024/1024):.1f} МБ</p>
        </div>
        """
        
        st.sidebar.markdown(performance_html, unsafe_allow_html=True)
    except ImportError:
        # Если psutil не установлен, просто пропускаем вывод информации
        pass
    except Exception as e:
        # Если произошла ошибка, выводим ее в консоль, но не в интерфейс
        print(f"Ошибка при получении информации о производительности: {str(e)}")

def show_app_version():
    """
    Отображает информацию о версии приложения
    """
    version_html = """
    <div style="text-align: center; margin-top: 20px; font-size: 0.8em; color: #757575;">
        <p>Версия 1.1.0 | Обновлено: 10.04.2025</p>
        <p>© 2025 Анализ и прогнозирование цен материалов</p>
    </div>
    """
    
    st.sidebar.markdown(version_html, unsafe_allow_html=True)