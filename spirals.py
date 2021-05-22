import cv2
import numpy as np
from numpy import array
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
    'speed':200,
    'major axis':5,
    'semi major axis':2,
    'angle':2,
    'constant':False,
    'colorspace':'BGR'
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

    if parameters['behavior'] == 'sin':
        push = sin(start * pi / 180)
    elif parameters['behavior'] == 'cos':
        push = cos(start * pi / 180)
    elif parameters['behavior'] == 'tan':
        push = tan(start * pi / 180)
    elif parameters['behavior'] == 'none':
        push = 0

    if parameters['constant'] == 'yes':
        push += (start * pi / 180)
    else:
        push += 0

    imageSize = 600

    origin = int(imageSize / 2)
    center = (origin,origin)
    numTurns = int(imageSize / lineThickness / 3)
    edge = lineThickness * numTurns
    image = np.full((imageSize,imageSize,3),[0,0,0],dtype=np.uint8)

    prev = None
    prev2 = None

    points = array([],dtype=np.int32)

    for i in range(0,360 * numTurns,inc):

        angle = i * pi / 180

        linear = i / 360

        r = linear * lineThickness

        pos = [int(origin + cos(angle) * r), int(origin + sin(angle) * r)]
        pos2 = [int(origin + cos(angle) * edge), int(origin + sin(angle) * edge)]

        bluchan = bluPhase + ((push + angle * bluDif) * 255 / pi)
        redchan = grnPhase + ((push + angle * grnDif) * 255 / pi)
        grnchan = redPhase + ((push + angle * redDif) * 255 / pi)

        if parameters['colorspace'] == 'HSV':
            bluchan %= 180
            redchan %= 255
            grnchan %= 255
        else:
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
            elif parameters['type'] == 'rectangle2':
                cv2.rectangle(image,prev2,pos2,color,lineThickness)
            elif parameters['type'] == 'circle':
                cv2.circle(image,pos,int(lineThickness/2),color,-1)
                cv2.circle(image,pos2,int(lineThickness/2),color,-1)
            elif parameters['type'] == 'ellipse':
                MA = parameters['major axis'] + 1
                SA = parameters['semi major axis'] + 1
                rotation = parameters['angle']
                distance = int(sqrt(abs(pos[0] - prev[0])**2 + abs(pos[1] - prev[1])**2))
                cv2.ellipse(image, (pos, (distance/MA,distance/SA),i + rotation), color, -1)
            elif parameters['type'] == 'polygon':
                points = np.append(points,pos)
                # points = np.append(points,pos2)

        prev = pos
        prev2 = pos2

    if parameters['type'] == 'polygon':
        points = points.reshape((-1,1,2))
        color = [bluPhase,grnPhase,redPhase]
        image = cv2.polylines(image,[points],False,color,2,5)

    if parameters['colorspace'] == 'HSV':
        image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)

    return image

def change(*args):
    parameters[args[1][0]] = args[1][1]
    pass

def check(*args):
    parameters[args[1][0]] = args[1][args[0] + 1]
    print(parameters[args[1][0]])
    pass

def createSliderWindow():

    cv2.namedWindow(windowName)

    inclimit = 359

    cv2.createTrackbar('inc',windowName,parameters['inc'],inclimit,lambda v:trackbarCallback('inc',v))

    thicknessLimit = 256

    cv2.createTrackbar('lineThickness',windowName,parameters['lineThickness'],thicknessLimit,lambda v:trackbarCallback('lineThickness',v))

    speedLimit = 700

    cv2.createTrackbar('speed',windowName,parameters['speed'],speedLimit,lambda v:trackbarCallback('speed',v))

    phaselimit = 256

    cv2.createTrackbar('blu Phase angle',windowName,parameters['blu Phase angle'],phaselimit,lambda v:trackbarCallback('blu Phase angle',v))
    cv2.createTrackbar('grn Phase angle',windowName,parameters['grn Phase angle'],phaselimit,lambda v:trackbarCallback('grn Phase angle',v))
    cv2.createTrackbar('red Phase angle',windowName,parameters['red Phase angle'],phaselimit,lambda v:trackbarCallback('red Phase angle',v))

    maxLimit = 500

    cv2.createTrackbar('blu Limit',windowName,parameters['blu Limit'],maxLimit,lambda v:trackbarCallback('blu Limit',v))
    cv2.createTrackbar('grn Limit',windowName,parameters['grn Limit'],maxLimit,lambda v:trackbarCallback('grn Limit',v))
    cv2.createTrackbar('red Limit',windowName,parameters['red Limit'],maxLimit,lambda v:trackbarCallback('red Limit',v))

    speedlimit = 1000

    cv2.createTrackbar('blu anglular dif',windowName,parameters['blu anglular dif'],speedlimit,lambda v:trackbarCallback('blu anglular dif',v))
    cv2.createTrackbar('grn anglular dif',windowName,parameters['grn anglular dif'],speedlimit,lambda v:trackbarCallback('grn anglular dif',v))
    cv2.createTrackbar('red anglular dif',windowName,parameters['red anglular dif'],speedlimit,lambda v:trackbarCallback('red anglular dif',v))

    axisLimit = 30

    cv2.createTrackbar('major axis (ellipse)',windowName,parameters['major axis'],axisLimit,lambda v:trackbarCallback('major axis',v))
    cv2.createTrackbar('semi major axis (ellipse)',windowName,parameters['semi major axis'],axisLimit,lambda v:trackbarCallback('semi major axis',v))

    maxAngle = 360

    cv2.createTrackbar('angle (ellipse)',windowName,parameters['angle'],maxAngle,lambda v:trackbarCallback('angle',v))

    cv2.createButton('spiral rect',change,['type','rectangle' ],2,0)
    cv2.createButton('edge rect'  ,change,['type','rectangle2'],2,0)
    cv2.createButton('lines'      ,change,['type','line'      ],2,1)
    cv2.createButton('circles'    ,change,['type','circle'    ],2,0)
    cv2.createButton('ellipse'    ,change,['type','ellipse'   ],2,0)
    cv2.createButton('polygon'    ,change,['type','polygon'   ],2,0)

    cv2.createButton('sin' ,change,['behavior','sin' ],2,1)
    cv2.createButton('cos' ,change,['behavior','cos' ],2,0)
    cv2.createButton('tan' ,change,['behavior','tan' ],2,0)
    cv2.createButton('none',change,['behavior','none'],2,0)

    cv2.createButton('HSV',change,['colorspace','HSV'],2,0)
    cv2.createButton('HSV',change,['colorspace','BGR'],2,1)

    cv2.createButton('direction',check,['direction','forward','reverse'],1,0)

    cv2.createButton('constant',check,['constant','yes','no'],1,0)

# hek
def trackbarCallback(windowName,value):
    parameters[windowName] = value

if __name__ == '__main__':
    windowName = 'spiral'
    createSliderWindow()
    start = 0
    while True:

        speed = parameters['speed']

        if parameters['direction'] == 'forward':
            start += speed
        elif parameters['direction'] == 'reverse':
            start -= speed
        im = spiral()
        cv2.imshow(windowName,im)
        cv2.waitKey(1)
