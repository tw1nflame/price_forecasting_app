import streamlit as st
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def apply_custom_css():
    """
    Применяет пользовательские CSS-стили к приложению
    """
    # Загружаем CSS из файла или задаем встроенные стили
    css = """
    /* Основные стили */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Заголовки */
    .main .block-container h1, .main .block-container h2 {
        color: #1E88E5;
        padding-bottom: 10px;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }
    
    .main .block-container h3, .main .block-container h4 {
        color: #0D47A1;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    /* Метрики */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: bold;
    }
    
    /* Боковая панель */
    [data-testid="stSidebar"] {
        background-color: #f5f5f5;
        padding: 1rem;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #1E88E5;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Информационные блоки */
    .stAlert {
        border-radius: 5px;
    }
    
    /* Кнопки */
    .stButton button {
        width: 100%;
    }
    
    /* Таблицы */
    [data-testid="stTable"] {
        width: 100%;
    }
    
    /* Графики */
    .js-plotly-plot {
        margin-bottom: 20px;
    }
    
    /* Разделители */
    hr {
        margin: 30px 0;
    }
    
    /* Подсказки */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #f0f0f0;
        color: #333;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    """
    
    # Применяем стили
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

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