# THIS SECTION OF THE COODE IS STILL IN PROGRESS





import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np

# model_dict = pickle.load(open('./model.p', 'rb'))
# 
# model = model_dict['model']

model  = tf.keras.models.load_model('model.h5')
cap = cv2.VideoCapture(2)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'L'}

#
# while True:
#     success, image = cap.read()
#     # print(image)
#     results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     image_height, image_width, _ = image.shape
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import mediapipe as mp
# import matplotlib.pyplot as plt
#
# cap = cv2.VideoCapture(0)
# # First step is to initialize the Hands class an store it in a variable
# mp_hands = mp.solutions.hands
#
# # Now second step is to set the hands function which will hold the landmarks points
# hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3,min_tracking_confidence=0.3)
#
# # Last step is to set up the drawing function of hands landmarks on the image
# mp_drawing = mp.solutions.drawing_utils
#
# while True:
#     success, image = cap.read()
#     # print(image)
#     results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     image_height, image_width, _ = image.shape
#
#     if results.multi_hand_landmarks:
#
#         for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
#             # print(f'{mp_hands.HandLandmark(20).name}:')
#             # print(f'x: {hand_landmarks.landmark[mp_hands.HandLandmark(20).value].x * image_width}')
#             # print(f'y: {hand_landmarks.landmark[mp_hands.HandLandmark(20).value].y * image_height}')
#             # print(f'z: {hand_landmarks.landmark[mp_hands.HandLandmark(20).value].z * image_width}n')
#
#             # wrist_x =int( hand_landmarks.landmark[mp_hands.HandLandmark(0).value].x*image_width)
#             wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark(0).value].y * image_height)
#             # middle_finger_tip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark(12).value].x * image_width)
#             middle_finger_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark(12).value].y * image_height)
#             little_finger_tip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark(20).value].x * image_width)
#             # little_finger_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark(20).value].y * image_height)
#
#             thumb_tip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark(4).value].x * image_width)
#             # thumb_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark(4).value].y * image_height)
#
#             # print(wrist_x,wrist_y,middle_finger_tip_x ,middle_finger_tip_y,thumb_tip_x ,thumb_tip_y)
#             cv2.rectangle(image, (little_finger_tip_x, middle_finger_tip_y), (thumb_tip_x, wrist_y), (255, 0, 0), 2)
#
#             # image1 =  little_finger_tip_x, middle_finger_tip_y, thumb_tip_x, wrist_y
#             img = image[little_finger_tip_x:middle_finger_tip_y, thumb_tip_x:wrist_y]
#
#             # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
#             # roi_image = gray[y:y + h, x:x + w]
#             # cv2.sav
#             print(img)
#
#             # cv2.imread(image1)
#
#
#             # print(image1)
#
#         mp_drawing.draw_landmarks(image=image, landmark_list=hand_landmarks,
#                                   connections=mp_hands.HAND_CONNECTIONS)
#         # image =
#
#         # x1 = 0
#         # y1 = 0
#         # x2 = 0
#         # y2 = 0
#         #
#         # cv2.rectangle(image, (thumb_tip_x, middle_finger_tip_x), (wrist_x, thumb_tip_x), (255,0,0), 2)
#
#     cv2.imshow("Output", image)
#     cv2.waitKey(1)
# #
# # import cv2
# # import mediapipe as mp
# #
# # mphands = mp.solutions.hands
# # hands = mphands.Hands()
# # mp_drawing = mp.solutions.drawing_utils
# # cap = cv2.VideoCapture(0)
# #
# # _, frame = cap.read()
# #
# # h, w, c = frame.shape
# #
# # while True:
# #     _, frame = cap.read()
# #     framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #     result = hands.process(framergb)
# #     hand_landmarks = result.multi_hand_landmarks
# #     if hand_landmarks:
# #         for handLMs in hand_landmarks:
# #             x_max = 0
# #             y_max = 0
# #             x_min = w
# #             y_min = h
# #             for lm in handLMs.landmark:
# #                 x, y = int(lm.x * w), int(lm.y * h)
# #                 if x > x_max:
# #                     x_max = x
# #                 if x < x_min:
# #                     x_min = x
# #                 if y > y_max:
# #                     y_max = y
# #                 if y < y_min:
# #                     y_min = y
# #             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
# #             mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
# #     cv2.imshow("Frame", frame)
# #
# #     cv2.waitKey(1)