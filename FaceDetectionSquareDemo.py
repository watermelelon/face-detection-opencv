import cv2
import mediapipe as mp
import time 

cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)


while True:
    success, image = cap.read()

    #transform the image to RGB from BGR
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    results = faceDetection.process(imageRGB)
    print(results)

    #we will get 6 points for each face detected
    if results.detections:
        for id, detection in enumerate(results.detections):
          
            #BUILT IN FUNCTION 
            #mpDraw.draw_detection(image, detection)
            
            
            #print (id, detection)
            #print (detection.score)
            #info of the bounding box, these points help to draw the box 
            #we can either draw it by ourselves or use the mediapipe function to draw it
            #print (detection.location_data.relative_bounding_box.xmin)
            #bounding box coming from the class C
            
            #DRAWING BY OURSELVES
            
            bboxC = detection.location_data.relative_bounding_box   
            ih, iw, ic = image.shape
            bbox = int (bboxC.xmin * iw), int (bboxC.ymin * ih), \
                     int (bboxC.width * iw), int (bboxC.height * ih)
            cv2.rectangle(image, bbox, (255,0,255), 2)
            cv2.putText(image, f'{int(detection.score[0]*100)}%',
                        (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                        2,(255,0,255),2)


   
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                 3,(0, 255, 0), 2)
    
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break


cv2.destroyAllWindows()