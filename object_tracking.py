import cv2
import numpy as np
from object_detection import ObjectDetection
import math

# Инициализируем обнаружение объектов
od = ObjectDetection()

# Наше видео
cap = cv2.VideoCapture("los_angeles.mp4")

# Инициализируем счетчик
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

# Воспроизведение кадров
while True:

    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Текущий кадр
    center_points_cur_frame = []

    # Обнаружение объектов в кадре
    (class_ids, scrores, boxes)= od.detect(frame)

    # Обводка объектов
    for box in boxes:
        (x, y, w, h) = box
        # Центры
        cx = int((x + x + w)/2)
        cy = int((y + y + w)/2)
        # Сохранение координат центров обьекта
        center_points_cur_frame.append((cx, cy))
        # Вывод номера и координат обьектов
        print('Номер Обьекта №',count, ' ', x, y, w, h)
        # Рисуем круг в центре объектов
        # cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        # Обводка обьектов в квадрат
        cv2.rectangle(frame, (x,y), (x + w, y+h), (0, 255, 0), 2)

    # Сравнение текущего кадра с предыдущим кадром
    if count <= 2:
        # Проходимся по текущим и прошлым объектам и если их дистанция мала, то это один обьект
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                # Считаем расстояние
                distans = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distans < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()
        for object_id, pt2 in tracking_objects_copy.items():
            object_exist = False
            for pt in center_points_cur_frame_copy:
                # Считаем расстояние
                distans = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                # Обновляем позицию ID
                if distans < 20:
                    tracking_objects[object_id] = pt
                    object_exist = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                        continue

            # Удаляем id которые были потряны
            if not object_exist:
                tracking_objects.pop(object_id)

        # Добавляем новые найденные ID
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1


    # Рисуем круг в центре обьекта и нумеруем его
    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)



    print('Tracking objects')
    print(tracking_objects)

    print('CURRENT FRAME LEFT PTS')
    print(center_points_cur_frame)

    print('PREV FRAME')
    print(center_points_prev_frame)

    cv2.imshow("Frame", frame)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


