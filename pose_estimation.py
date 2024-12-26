import cv2
import numpy as np
import matplotlib.pyplot as plt


BODY_PARTS={"Nose":0,"Neck":1,"RShoulder":2,"RElbow":3,"RWrist":4,
            "LShoulder":5,"LElbow":6,"LWrist":7,"RHip":8,"RKnee":9,
            "RAnkle":10,"LHip":11,"LKnee":12,"LAnkle":13,"REye":14,
            "LEye":15,"REar":16,"LEar":17,"Background":18}

            
POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"],
    ["Neck", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"],
    ["Neck", "Nose"], ["Nose", "REye"], ["REye", "REar"],
    ["Nose", "LEye"], ["LEye", "LEar"]
]

width = 368
height = 368
inWidth = width 
inHeight = height 

# Check if the file exists
import os
if not os.path.exists("C:\\Users\\SATYA PRAKASH\\OneDrive\\Desktop\\New folder\\graph_opt (1).pb"):
    print("Error: Model file not found at /content/graph_opt.pb")
    # Download or move the model file to the correct location
else:
    # Attempt to load the model
    try:
        net = cv2.dnn.readNetFromTensorflow("C:\\Users\\SATYA PRAKASH\\OneDrive\\Desktop\\New folder\\graph_opt (1).pb")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Possible reasons: file corrupted, incompatible model, or permissions issue.")
        # Further troubleshooting steps, like checking OpenCV version or file permissions


thresh = 0.2

def poseDetector(frame):
    framewidth = frame.shape[1] 
    frameHeight = frame.shape[0] 
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward() 
    out = out[:, :19, :, :] 
    assert(len(BODY_PARTS) <= out.shape[1])
    points = []
    for i in range(len(BODY_PARTS)):
       
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (framewidth * point[0]) / out.shape[3] 
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thresh else None) 
    
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        
        # Fix: Changed 'Lshoulder' to 'LShoulder' in BODY_PARTS to match POSE_PAIRS
        # This ensures that the assertion passes and the code executes correctly.
        
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)
        
        #The code below was not indented correctly, causing the IndentationError.
        #Indenting the lines within the for loop fixes the issue.
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        # Draw lines and points only if both points are detected
        if points[idFrom] and points[idTo]:
            # Draw a line connecting the two points
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)

            # Draw circles (ellipses) at the two points
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    t, _ = net.getPerfProfile()
    # Return the processed frame
    return frame

# Example usage
input = cv2.imread("C:\\Users\\SATYA PRAKASH\\Downloads\\scale_1200.jpg")
output = poseDetector(input)

cv2.imwrite("Output-image.png",output)

cv2.waitKey(0)
cv2.destroyAllWindows()
