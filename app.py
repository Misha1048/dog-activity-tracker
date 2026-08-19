import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime
from collections import deque
from ultralytics import YOLO

# 1. Настройка главной страницы
st.set_page_config(page_title="Dog Activity Tracker", page_icon="🐾", layout="wide")

st.title("🐾 Умная камера-няня: Трекер активности Лилы")
st.markdown("Система распознавания и отслеживания активности домашней собаки с использованием алгоритмов машинного обучения.")
st.markdown("---")

# Кэшируем загрузку модели
@st.cache_resource
def load_model():
    return YOLO('runs/pose/lila_pose_model/weights/best.pt')

# 2. Разделяем экран на две колонки
col_video, col_log = st.columns([2, 1])

with col_video:
    st.subheader("🔴 Прямой эфир")
    video_placeholder = st.empty()

with col_log:
    st.subheader("Текущий статус")
    status_placeholder = st.empty()
    status_placeholder.markdown("### 🟡 Ожидание данных...")
    st.markdown("---")
    st.subheader("Журнал событий")
    history_placeholder = st.empty()
    history_df = pd.DataFrame(columns=["Время", "Действие"])
    history_placeholder.dataframe(history_df, use_container_width=True, hide_index=True)

# 3. Панель управления
st.sidebar.header("⚙️ Управление")

st.sidebar.subheader("🎥 Источник видео")
camera_source = st.sidebar.radio("Выберите тип камеры:", ("IP-камера (TAPO / RTSP)", "Веб-камера (USB)"))

rtsp_url = ""
if camera_source == "IP-камера (TAPO / RTSP)":
    rtsp_url = st.sidebar.text_input("Введите RTSP ссылку:", value="rtsp://misha1998:ghost1048@192.168.1.29:554/stream1")
    st.sidebar.caption("Формат TAPO: rtsp://[логин]:[пароль]@[IP_адрес]:554/stream1")

run_camera = st.sidebar.checkbox("▶️ Включить трансляцию")

st.sidebar.markdown("---")
st.sidebar.markdown("**Модель:** YOLOv8-Pose (Кастомная)")
st.sidebar.markdown("**Объект:** Французский бульдог")

# 4. ГЛАВНЫЙ ЦИКЛ ОБРАБОТКИ ВИДЕО
if run_camera:
    model = load_model()
    
    if camera_source == "Веб-камера (USB)":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(rtsp_url) 
        
    history_length = 10 
    y_coords_history = deque(maxlen=history_length)
    scratch_cooldown = 0
    
    last_logged_status = None
    log_data = []
    
    while cap.isOpened() and run_camera:
        ret, frame = cap.read()
        if not ret:
            video_placeholder.error("Не удалось подключиться к потоку. Проверьте RTSP ссылку и сеть!")
            break
            
        results = model(frame, conf=0.3, verbose=False)
        
        annotated_frame = frame.copy()
        current_status = "ИЩУ СИЛУЭТ..."
        
        best_dog_points = None
        frame_height = frame.shape[0] 
        lowest_y = 0 
        
        # --- ЗОНА КОРМЛЕНИЯ ---
        feed_zone = (360, 570, 470, 670) 
        cv2.rectangle(annotated_frame, (feed_zone[0], feed_zone[1]), (feed_zone[2], feed_zone[3]), (0, 255, 255), 2)
        cv2.putText(annotated_frame, "Feeding Zone", (feed_zone[0], feed_zone[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # --- ФИЛЬТРАЦИЯ ОТ ЛОЖНЫХ СРАБАТЫВАНИЙ ---
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            for i in range(len(results[0].keypoints.xy)):
                keypoints = results[0].keypoints.xy[i].cpu().numpy()
                valid_points = [kp for kp in keypoints if kp[0] != 0 and kp[1] != 0]
                
                if len(valid_points) > 0:
                    valid_points = np.array(valid_points)
                    min_x, min_y = np.min(valid_points, axis=0)
                    max_x, max_y = np.max(valid_points, axis=0)
                    
                    # Находим центр текущего силуэта
                    center_x = (min_x + max_x) / 2
                    center_y = (min_y + max_y) / 2
                    
                    # Проверяем, попал ли силуэт в желтую зону миски
                    in_feed_zone = (feed_zone[0] < center_x < feed_zone[2]) and (feed_zone[1] < center_y < feed_zone[3])
                    
                    # Если силуэт НЕ в зоне кормления, применяем фильтры
                    if not in_feed_zone:
                        ratio = (max_x - min_x) / ((max_y - min_y) + 0.001)
                        if ratio < 0.60:
                            continue 
                        if min_y < frame_height * 0.30:
                            continue
                        if (max_x - min_x) < frame.shape[1] * 0.05:
                            continue
                            
                    # Фильтр 3: Берем самый нижний силуэт из оставшихся
                    if max_y > lowest_y:
                        lowest_y = max_y
                        best_dog_points = valid_points
        
        # --- МАТЕМАТИКА АКТИВНОСТИ И РУЧНАЯ ОТРИСОВКА ---
        if best_dog_points is not None:
            min_x, min_y = np.min(best_dog_points, axis=0)
            max_x, max_y = np.max(best_dog_points, axis=0)
            
            # Отрисовка Лилы
            cv2.rectangle(annotated_frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 50, 50), 3)
            cv2.putText(annotated_frame, "Lila", (int(min_x), int(min_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 50, 50), 2)
            
            for kp in best_dog_points:
                cv2.circle(annotated_frame, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)

            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            ratio = (max_x - min_x) / ((max_y - min_y) + 0.001)
            
            # Проверка зоны кормления
            if (feed_zone[0] < center_x < feed_zone[2]) and (feed_zone[1] < center_y < feed_zone[3]):
                base_status = "ЕСТ 🍖"
            elif ratio > 1.5:
                base_status = "ЛЕЖИТ 💤"
            elif ratio < 0.85:
                base_status = "СИДИТ 🐕"
            else:
                base_status = "СТОИТ 🐾"

            y_coords_history.append(best_dog_points[:, 1])
            
            if len(y_coords_history) == history_length:
                history_array = np.array(y_coords_history)
                amplitudes = np.max(history_array, axis=0) - np.min(history_array, axis=0)
                max_amplitude = np.max(amplitudes)
                
                if max_amplitude > 40:
                    scratch_cooldown = 15

            if scratch_cooldown > 0:
                current_status = "ЧЕШЕТСЯ! ⚡"
                scratch_cooldown -= 1
            else:
                current_status = base_status 

        # --- ОБНОВЛЕНИЕ ЖУРНАЛА СОБЫТИЙ ---
        if current_status != "ИЩУ СИЛУЭТ..." and current_status != last_logged_status:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            log_data.insert(0, {"Время": current_time, "Действие": current_status})
            
            if len(log_data) > 15:
                log_data.pop()
                
            history_df = pd.DataFrame(log_data)
            history_placeholder.dataframe(history_df, use_container_width=True, hide_index=True)
            
            last_logged_status = current_status
        # --- ОБНОВЛЕНИЕ ИНТЕРФЕЙСА ---
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
        
        if current_status == "ИЩУ СИЛУЭТ...":
            status_placeholder.markdown(f"### 🟡 {current_status}")
        elif "ЧЕШЕТСЯ" in current_status:
            status_placeholder.markdown(f"### 🔴 {current_status}")
        else:
            status_placeholder.markdown(f"### 🟢 {current_status}")
            
    cap.release()
else:
    video_placeholder.info("Нажмите галочку в меню слева, чтобы запустить камеру.")