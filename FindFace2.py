import cv2
import mediapipe as mp
import numpy as np
# Инициализация объектов
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)


def landmarks_to_numpy(landmarks):
    arr_xyz = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
    print(arr_xyz)
    return arr_xyz
    # comment for test
    # np.savez(file_name, *nparrays) or np.save(file_name, np_array)
    # np.load(file_name)

def save_landmarks_to_numpy(image, landmarks):
    pass

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        success, image = cap.read()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = True

        results = face_mesh.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                # save(landmarks, image)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())

        # Display the image
        cv2.imshow('MediaPipe FaceMesh', image)

        # Terminate the process
        if cv2.waitKey(100) & 0xFF == 27:
            break
cap.release()



