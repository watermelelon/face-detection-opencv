from turtle import color
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import mediapipe as mp

#FACE MESH (download the specific library, model)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh() 

#Video Capture
#We want to keep consistency in the video size
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1) #we can detect up to 2 faces, we can change it to 1 or more than 2
plotY = LivePlot(640, 480, [10, 40],invert=True) #we will plot the ratio of the eye opening, we want to see it between 0 and 1, we want to invert it because when the eye is closed the ratio is smaller than when the eye is open


pointList = [22,23,24, 26, 110, 157, 158, 159, 160, 161, 130, 243] #these are the points we can find in the face
ratioList = []
blinkCounter = 0
counter = 0
color = (255,0,255)


while True:

    ### ALTERNATIVE: makign sure the video runs on loop
    #if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
    #   cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


    #Static images
    ret, image = cap.read()

    if not ret:
        break
    height, width, _ = image.shape
    #print (f"height={height}, width={width}")

    image, faces = detector.findFaceMesh(image,draw=False) #we can draw the points on the image or not, we will do it ourselves, so we set it to false

    if faces:
        faces = faces[0]  # Get the first detected face
        for id in pointList:
            cv2.circle(image, (faces[id][0], faces[id][1]), 3, color, cv2.FILLED)

        leftUp = faces[159]
        leftDown = faces[145]
        leftLeft = faces[130]
        leftRight = faces[243]
        lengthVer, _ = detector.findDistance(leftUp, leftDown)
        lengthHor, _ = detector.findDistance(leftLeft, leftRight)
        cv2.line(image, leftUp, leftDown, (0, 200, 0), 2)
        cv2.line(image, leftLeft, leftRight, (0, 200, 0), 2)
        
        ratio = (lengthVer/lengthHor*100)
        ratioList.append(ratio)
        #Ratio list for smoothing the plot 
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList)/len(ratioList)

        if ratioAvg < 25 and counter == 0:
            blinkCounter += 1
            color = (0,200,0)
            counter = 1
        
        #to avoid counting multiple blinks for one blink, we will use a counter to count the number of frames that the eye is closed, if the counter is greater than 10, we will reset it to 0, this
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (255,0,255)

        cvzone.putTextRect(image, f'Blink Count: {blinkCounter}', (50, 100), scale=2, thickness=2, offset=10, colorR= color ) #we want to put the text on the image, we want to put it at the top left corner, we want to scale it, we want to make it thicker, we want to add some offset to make it look better


        imgPlot = plotY.update(ratioAvg, color=color) #we want to update the plot with the new ratio value
        imgPlot = cv2.resize(imgPlot, (640, 480))
        #cv2.imshow("Plot", imgPlot)
        imageStack = cvzone.stackImages([image, imgPlot], 2, 1) #we to stack the images vertically, we want to have 2 rows and 1 column

    else:
        imgPlot = cv2.resize(imgPlot, (640, 480))
        #cv2.imshow("Plot", imgPlot)
        imageStack = cvzone.stackImages([image, image], 2, 1) #we want to stack the images vertically, we want to have 2 rows and 1 column






    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)

    ###   ALTERNATIVE, MANUAL DRAWING OF THE FACE LANDMARKS
    # for face_landmarks in result.multi_face_landmarks:
    #     for i in range(0,468):
    #         pt = face_landmarks.landmark[i]  # Example: landmark index 0
    #         x = int(pt.x * width)
    #         y = int(pt.y * height)

    #         #Once we have the coordinates of the landmarks as integers, we can use them as coordinates 
    #         #to draw the circles on the picture 
    #         cv2.circle(image, (x, y), 2, (100, 100, 0), -1)
            
    #print (result)


    cv2.imshow("Original Image", imageStack)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break #waitkey 0 -> freezes the frame
                   #waitkey 1 -> continues the video

