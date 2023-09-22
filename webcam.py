import cv2
from utils import *
from keras.models import load_model
import numpy as np



# Detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.4
res = [0.7,0.2,0.1]

model = load_model("model/signdetection.h5")

colors = [(245,117,16),(117,245,16),(16,117,245)]
actions = np.array(["hello","thanks","iloveyou"])

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_tracking_confidence=0.5,min_detection_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read frame
        ret,frame = cap.read()

        # Make detections
        image,results = mediapipe_detection(frame,holistic)

        # Draw landmarks
        draw_landmarks(image,results)

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence)==30:
            res = model.predict(np.expand_dims(sequence,axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

            if np.unique(predictions[-10:])[0]==np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence)>0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])


        cv2.rectangle(image,(0,0),(640,40),(245,117,16),-1)
        cv2.putText(image," ".join(sentence),(3,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        
        if len(sentence)>5:
            sentence=sentence[-5:]
        
        # visualizing prob
        image = prob_viz(res,actions,image,colors)

        # Show to screen
        cv2.imshow("Opencv feed",image)

        # Break
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()