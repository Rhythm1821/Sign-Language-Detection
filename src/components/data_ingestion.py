import cv2
import numpy as np
import sys,os


sys.path.append(".")
from utils import *

class DataIngestion:
    def __init__(self):
        self.data_path = os.path.join('notebook','MP_DATA')
        self.actions = np.array(["hello","thanks","iloveyou"])
        self.num_sequences = 30
        self.sequence_length = 30
    
    def make_dir(self):
        for action in self.actions:
            for sequence in range(self.num_sequences):
                try:
                    os.makedirs(os.path.join(self.data_path,action,str(sequence)),exist_ok=True)
                except Exception as e:
                    print(e)

    def collect_data(self):
        cap = cv2.VideoCapture(0)
        # labelling the actions

        with mp_holistic.Holistic(min_tracking_confidence=0.5,min_detection_confidence=0.5) as holistic:

            # Loop through actions
            for action in self.actions:
                # Loop through sequences i.e. videos
                for sequence in range(self.num_sequences):
                    # Loop through video length i.e. video length
                    for frame_num in range(self.sequence_length):
                        # Read frame
                        ret, frame = cap.read()

                        # Make detections
                        image,results = mediapipe_detection(frame,holistic)

                        # Draw landmarks
                        draw_styled_landmarks(image,results)

                        # Collection breaks
                        if frame_num == 0: 
                            cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(2000)
                        else: 
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                        
                        # Show to screen
                        cv2.imshow("Opencv feed",image)
                        
                        # Export keypoints
                        keypoints=extract_keypoints(results)
                        npy_path = os.path.join(self.data_path,action,str(sequence),str(frame_num))
                        np.save(npy_path,keypoints)

                    

                        if cv2.waitKey(10) & 0xFF==ord('q'):
                            break

            cap.release()
            cv2.destroyAllWindows()
  
    def store_data(self):
        try:
            label_map = {label:num for num,label in enumerate(self.actions)}
            sequences,labels=[],[]
            for action in self.actions:
                for sequence in range(self.num_sequences):
                    window = []
                    for frame_num in range(self.sequence_length):
                        res=np.load(os.path.join(self.data_path,action,str(sequence),f"{frame_num}.npy"))
                        window.append(res)
                    sequences.append(window)
                    labels.append(label_map[action])
            return sequences, labels
        except Exception as e:
            print("Failed storing the data:",e)            
    
    




        

