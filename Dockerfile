# Используем официальный образ Python 3.11 как базовый
FROM python:3.11-slim

# Метаданные о создателе образа
LABEL maintainer="Price Forecasting App Team"

# Устанавливаем переменные среды
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл с зависимостями Python
COPY requirements.txt .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы проекта
COPY . .

# Делаем порт 8501 доступным извне контейнера (стандартный порт Streamlit)
EXPOSE 8501

# Запускаем приложение
CMD ["streamlit", "run", "app.py"]

# Примечание по использованию:
# Для запуска контейнера с указанными ресурсами используйте команду:
# docker run -d -p 8501:8501 --memory="16g" --cpus="14" --mount source=app_data,target=/app/data price_forecasting_app
# 
# Где:
# --memory="16g" - выделение 16 ГБ оперативной памяти
# --cpus="14" - использование 14 ядер CPU
# --mount source=app_data,target=/app/data - монтирование тома для хранения данных (50 ГБ)
# 
# Перед запуском создайте том для данных:
# docker volume create app_data
# 
# Для задания размера тома используйте опции драйвера (зависит от используемой системы).
# Например, в Docker Desktop с WSL2 на Windows: 
# 1. Настройте размер диска WSL2 через %USERPROFILE%\.wslconfig
# 2. Установите параметр size=50GB в секции [wsl2]
