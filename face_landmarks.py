import cv2

class facelm():
    import mediapipe as mp
    def __init__(self,window_width,window_height):
        self.window_height = window_height
        self.window_width = window_width
        self.face = self.mp.solutions.face_mesh.FaceMesh(False,2,.5,.5)
        self.draw_face = self.mp.solutions.drawing_utils

    def landmarks(self,frame,colour = (0,255,0)):
        self.drawSpecC = self.draw_face.DrawingSpec(thickness = 0,circle_radius =0, color = (255,255,255))
        self.drawSpecL = self.draw_face.DrawingSpec(thickness = 2,circle_radius =1, color = colour)
        landmarks = []
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face.process(frameRGB)
        if (results.multi_face_landmarks) != None:
            for lms in results.multi_face_landmarks:
                self.draw_face.draw_landmarks(frame,lms,self.mp.solutions.face_mesh.FACE_CONNECTIONS, self.drawSpecC,self.drawSpecL)
                landmark = []
                for lm in lms.landmark:
                    landmark.append((int(self.window_width*lm.x),int(self.window_height*lm.x)))
                landmarks.append(landmark)
        return landmarks
import mediapipe as mp