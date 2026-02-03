import cv2
import mediapipe as mp

#FACE MESH (download the specific library, model)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh() 

#Video Capture
cap = cv2.VideoCapture(0)

while True:

    #Static images
    ret, image = cap.read()
    if not ret:
        break
    height, width, _ = image.shape
    #print (f"height={height}, width={width}")


    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)

    for face_landmarks in result.multi_face_landmarks:
        for i in range(0,468):
            pt = face_landmarks.landmark[i]  # Example: landmark index 0
            x = int(pt.x * width)
            y = int(pt.y * height)

            #Once we have the coordinates of the landmarks as integers, we can use them as coordinates 
            #to draw the circles on the picture 
            cv2.circle(image, (x, y), 2, (100, 100, 0), -1)
            
    #print (result)


    cv2.imshow("Original Image", image)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break #waitkey 0 -> freezes the frame
                   #waitkey 1 -> continues the video

