from typing import List, Dict, Tuple
import streamlit as st
from megadetector.utils import url_utils
from megadetector.visualization import visualization_utils as vis_utils
from megadetector.detection import run_detector
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd

# Путь к файлам модели и классов
CLASSIFICATION_MODEL_PATH = "src/nto_sochi_model.pt"
CLASSES_DICT: Dict[int, str] = pd.read_csv("src/classes.csv").set_index("id")["species"].to_dict()


class ImageClassifier:
    def __init__(self, model_path: str, classes_dict: Dict[int, str]):
        self.model = torch.jit.load(model_path)
        self.classes_dict = classes_dict
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def classify(self, image: Image.Image) -> Tuple[int, float]:
        image_tensor = self.preprocess(image).unsqueeze(0)
        predictions = self.model(image_tensor)
        probabilities = F.softmax(predictions, dim=1)
        predicted_class_index = np.argmax(predictions[0].detach().numpy())
        confidence = round(probabilities[0, predicted_class_index].item(), 2)
        return predicted_class_index, confidence


class MegaDetectorApp:
    def __init__(self, classifier: ImageClassifier, detection_model_path: str):
        self.classifier = classifier
        self.detection_model = run_detector.load_detector(detection_model_path)

    def crop_and_classify(self, image: Image.Image, detections: List[Dict]) -> None:
        image_width, image_height = image.size
        for idx, detection in enumerate(detections):
            x, y, w, h = detection["bbox"]
            x1, y1, x2, y2 = (
                int(x * image_width),
                int(y * image_height),
                int((x + w) * image_width),
                int((y + h) * image_height),
            )
            cropped_image = image.crop((x1, y1, x2, y2))
            if cropped_image.size[0] > 0 and cropped_image.size[1] > 0:
                detection["category"], detection["conf"] = self.classifier.classify(
                    cropped_image
                )

    def detect_objects(self, image: Image.Image, threshold: float) -> Tuple[List[Dict], Dict]:
        result: Dict = self.detection_model.generate_detections_one_image(
            image, detection_threshold=threshold
        )
        detections_above_threshold: List[Dict] = [
            d
            for d in result["detections"]
            if d["conf"] > threshold and d["category"] == str(1)
        ]
        self.crop_and_classify(image, detections_above_threshold)
        return detections_above_threshold, result

    def render_detections(self, image: Image.Image, detections: List[Dict]) -> None:
        vis_utils.render_detection_bounding_boxes(
            detections=detections, image=image, label_map=CLASSES_DICT
        )


# Интерфейс Streamlit
st.title("🐾 MegaDetector Животные 🐾")
st.markdown(
    (
        """
Добро пожаловать в **MegaDetector**! Это приложение позволяет загружать изображения и находить животных с
использованием модели MegaDetector.
Выберите источник изображения ниже, чтобы начать.
"""
    )
)

upload_method = st.selectbox(
    "Выберите источник изображения", ["URL", "Загрузка с компьютера"], index=0
)
threshold = st.slider("Установите порог вероятности", 0.0, 1.0, 0.2, 0.05)
st.write(f"Используется порог: {threshold}")

if upload_method == "URL":
    st.subheader("Шаг 1: Введите URL изображения")
    image_url = st.text_input("Введите URL", "")
    if not image_url:
        st.error("Пожалуйста, введите URL изображения.")
    else:
        if st.button("Загрузить изображение"):
            with st.spinner("Загрузка изображения..."):
                temporary_filename = url_utils.download_url(image_url)
                image = vis_utils.load_image(temporary_filename)
                st.image(image, caption="Загруженное изображение", use_container_width=True)
            with st.spinner("Обработка детекции..."):
                # Инициализация приложения
                classifier = ImageClassifier(CLASSIFICATION_MODEL_PATH, CLASSES_DICT)
                app = MegaDetectorApp(classifier, "MDV5A")
                detections, _ = app.detect_objects(image, threshold)
                st.write(f"Найдено {len(detections)} объектов с порогом выше {threshold}")
                app.render_detections(image, detections)
                st.image(image, caption="Изображение с детекцией", use_container_width=True)

elif upload_method == "Загрузка с компьютера":
    st.subheader("Шаг 1: Загрузите изображение с вашего компьютера")
    uploaded_file = st.file_uploader(
        "Выберите изображение", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        with st.spinner("Загрузка изображения..."):
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженное изображение", use_container_width=True)
        with st.spinner("Обработка детекции..."):
            # Инициализация приложения
            classifier = ImageClassifier(CLASSIFICATION_MODEL_PATH, CLASSES_DICT)
            app = MegaDetectorApp(classifier, "MDV5A")
            detections, _ = app.detect_objects(image, threshold)
            st.write(f"Найдено {len(detections)} объектов с порогом выше {threshold}")
            app.render_detections(image, detections)
            st.image(image, caption="Изображение с детекцией", use_container_width=True)
