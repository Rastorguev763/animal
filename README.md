# MegaDetector Image Detection with Streamlit

Этот проект использует модель MegaDetector для детекции объектов на изображениях и визуализирует результаты с помощью Streamlit. Вы можете загружать изображения, проводить детекцию объектов, а затем отображать результаты в интерактивном веб-интерфейсе.

## Требования

Для работы проекта необходимы следующие зависимости:

- Python 3.11
- MegaDetector
- Streamlit
- Poetry

## Установка

1. Клонируйте репозиторий:

   ```bash
   git clone https://github.com/Rastorguev763/animal.git
   cd animal
   ```

2. Установите Python 3.11
3. Установите Poetry

    ```bash
    pip install poetry
    ```

4. Установите зависимости через poetry

    ```bash
    poetry install
    ```

5. Для запуска используйте команду

    ```bash
    poetry run streamlit run src/main_app.py
    ```
