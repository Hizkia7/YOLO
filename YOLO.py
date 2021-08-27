# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 19:44:15 2021

@author: User
"""
import numpy as np
import cv2

classes = []

with open('coco.names', 'r') as f:
    classes=f.read().splitlines()

net = cv2.dnn.readNet('yolov3-spp.weights', 'yolov3-spp.cfg')

img = cv2.imread('img.jpeg')
#vid=cv2.
height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), 
                             (0,0,0), swapRB=True, 
                             crop=False)

net.setInput(blob)
layersNames = net.getLayerNames()
output_layer=[layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#output_layer=net.getUnconnectedOutLayersNames()
layer_outputs=net.forward(output_layer)


boxes=[]
confidences=[]
class_idx=[]
for output in layer_outputs:
    for detection in output:
        scores=detection[5:]
        max_idx=np.argmax(scores)
        confidence=scores[max_idx]
        if confidence > 0.5:
            center_x = int(detection[0]*width)
            center_y= int(detection[1]*height)
            
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            
            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_idx.append(max_idx)
            
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

print(class_idx)

font = cv2.FONT_HERSHEY_PLAIN

color = np.random.uniform(0, 255, size=(len(boxes),3))

print(len(indexes))

for i in indexes.flatten():
    x,y,w,h=boxes[i]
    label=str(classes[class_idx[i]])
    confidence=str(round(confidences[i],2))
    colors=color[i]
    cv2.rectangle(img, (x,y), (x+w, y+h), colors, 2)
    cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()            
            
