import cv2
import numpy as np
from math import sin, cos, tan, pi, sqrt

# defaults
parameters = {
    'lineThickness':10,
    'blu Phase angle':50,
    'grn Phase angle':90,
    'red Phase angle':100,
    'blu Limit':190,
    'grn Limit':90,
    'red Limit':100,
    'blu anglular dif':1,
    'grn anglular dif':30,
    'red anglular dif':65,
    'inc':0,
    'type':'line',
    'behavior':'sin',
    'direction':'forward',
    'speed':200
}

def spiral():

    # read parameters

    lineThickness = parameters['lineThickness'] + 1

    bluPhase = parameters['blu Phase angle']

    grnPhase = parameters['grn Phase angle']
    redPhase = parameters['red Phase angle']

    bluLimit = parameters['blu Limit'] + 1
    grnLimit = parameters['grn Limit'] + 1
    redLimit = parameters['red Limit'] + 1

    bluDif = parameters['blu anglular dif']
    grnDif = parameters['grn anglular dif']
    redDif = parameters['red anglular dif']

    inc = parameters['inc'] + 1

    speed = parameters['speed']

    if parameters['behavior'] == 'sin':
        push = sin(start * pi / 180) * speed
    elif parameters['behavior'] == 'cos':
        push = cos(start * pi / 180) * speed
    elif parameters['behavior'] == 'tan':
        push = tan(start * pi / 180) * speed
    elif parameters['behavior'] == 'linear':
        push = start * int(speed / 30)

    imageSize = 600

    origin = int(imageSize / 2)
    center = (origin,origin)
    numTurns = int(imageSize / lineThickness / 3)
    edge = lineThickness * numTurns
    image = np.full((imageSize,imageSize,3),[0,0,0],dtype=np.uint8)

    prev = None

    for i in range(0,360 * numTurns,inc):

        angle = i * pi / 180
        linear = i / 360

        r = linear * lineThickness

        pos = int(origin + cos(angle) * r), int(origin + sin(angle) * r)
        pos2 = int(origin + cos(angle) * edge), int(origin + sin(angle) * edge)

        bluchan = push + bluPhase + (angle*bluDif)
        redchan = push + grnPhase + (angle*grnDif)
        grnchan = push + redPhase + (angle*redDif)

        bluchan %= bluLimit
        redchan %= grnLimit
        grnchan %= redLimit

        color = [bluchan,redchan,grnchan]

        if prev != None:
            if parameters['type'] == 'line':
                cv2.line(image,prev,pos,color,lineThickness)
                cv2.line(image,prev2,pos2,color,lineThickness)
            elif parameters['type'] == 'rectangle':
                cv2.rectangle(image,prev,pos,color,lineThickness)
                cv2.rectangle(image,prev2,pos2,color,lineThickness)

        prev = pos
        prev2 = pos2

    return image

def behavior(*args):
    print(args[1])
    parameters['behavior'] = args[1]
    pass

def type(*args):
    print(args[1])
    parameters['type'] = args[1]
    pass

def direction(*args):
    print(args[0])
    if args[0]:
        parameters['direction'] = 'forward'
    else:
        parameters['direction'] = 'reverse'
    pass

def createSliderWindow():

    cv2.namedWindow(windowName)

    thicknessLimit = 256

    cv2.createTrackbar(key := 'lineThickness',windowName,parameters[key],thicknessLimit,lambda v:trackbarCallback('lineThickness',v))

    phaselimit = 256

    cv2.createTrackbar(key := 'blu Phase angle',windowName,parameters[key],phaselimit,lambda v:trackbarCallback('blu Phase angle',v))
    cv2.createTrackbar(key := 'grn Phase angle',windowName,parameters[key],phaselimit,lambda v:trackbarCallback('grn Phase angle',v))
    cv2.createTrackbar(key := 'red Phase angle',windowName,parameters[key],phaselimit,lambda v:trackbarCallback('red Phase angle',v))

    maxLimit = 500

    cv2.createTrackbar(key := 'blu Limit',windowName,parameters[key],maxLimit,lambda v:trackbarCallback('blu Limit',v))
    cv2.createTrackbar(key := 'grn Limit',windowName,parameters[key],maxLimit,lambda v:trackbarCallback('grn Limit',v))
    cv2.createTrackbar(key := 'red Limit',windowName,parameters[key],maxLimit,lambda v:trackbarCallback('red Limit',v))

    speedlimit = 500

    cv2.createTrackbar(key := 'blu anglular dif',windowName,parameters[key],speedlimit,lambda v:trackbarCallback('blu anglular dif',v))
    cv2.createTrackbar(key := 'grn anglular dif',windowName,parameters[key],speedlimit,lambda v:trackbarCallback('grn anglular dif',v))
    cv2.createTrackbar(key := 'red anglular dif',windowName,parameters[key],speedlimit,lambda v:trackbarCallback('red anglular dif',v))

    inclimit = 359

    cv2.createTrackbar(key := 'inc',windowName,parameters[key],inclimit,lambda v:trackbarCallback('inc',v))

    speedLimit = 500

    cv2.createTrackbar(key := 'speed',windowName,parameters[key],speedLimit,lambda v:trackbarCallback('speed',v))

    cv2.createButton('rectangles',type,'rectangle',2,0)
    cv2.createButton('lines',type,'line',2,1)

    cv2.createButton('sin',behavior,'sin',2,1)
    cv2.createButton('cos',behavior,'cos',2,0)
    cv2.createButton('tan',behavior,'tan',2,0)
    cv2.createButton('linear',behavior,'linear',2,0)

    cv2.createButton('direction',direction,'linear',1,0)

# hek
def trackbarCallback(windowName,value):
    print(windowName,value,end='{:>20s}'.format('\r'))
    parameters[windowName] = value

if __name__ == '__main__':
    windowName = 'spiral'
    createSliderWindow()
    start = 0
    while True:

        if parameters['direction'] == 'forward':
            start += 1
        elif parameters['direction'] == 'reverse':
            start -= 1
        im = spiral()
        cv2.imshow(windowName,im)
        cv2.waitKey(1)
