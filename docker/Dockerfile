FROM python:3.11-slim

# Установка переменных окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Обновляем пакеты и устанавливаем необходимые зависимости для работы с Python
RUN apt-get update && apt-get install -y\
    python3-dev \
    gcc \
    libc-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip и устанавливаем Poetry
RUN pip install --upgrade pip setuptools poetry

# Установка рабочего каталога
WORKDIR /usr/src/app

# Копирование всех файлов проекта
COPY ./src /usr/src/app/
COPY ./poetry.lock /usr/src/app/poetry.lock
COPY ./pyproject.toml /usr/src/app/pyproject.toml
# Установка зависимостей через Poetry
ENV POETRY_HTTP_TIMEOUT=3600
RUN poetry install --only main --no-root --no-cache -v

# Открываем порт для приложения
EXPOSE 8501

# Запуск приложения через Streamlit
CMD ["poetry", "run", "streamlit", "run", "src/main_app.py"]
