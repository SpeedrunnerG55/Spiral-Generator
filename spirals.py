import cv2
import numpy as np
from numpy import array
from math import sin, cos, tan, pi, sqrt
from time import time_ns

# defaults
parameters = {
    'imageSize':500,
    'graph':{'fill':True,'phase':False,'spiral':False,'unit':False,'time':False},
    'colorspace':'BGR',
    'clock':100,
    'timeAngle':None,
    'lineThickness':20,
    'inc':0,
    'type':'polygon',
    'majr axis':0,
    'semi axis':0,
    'angle':0,
    'phase angle':[0,0,0],
    'corse angle dif':[4,0,0],
    'fine angle dif':[180,180,180],
    'trig':['none','none','none'],
    'direction':['fwd','fwd','fwd'],
    'fine speed':[0,0,0],
    'corse speed':[0,0,0],
    'constant':[False,False,False],
    'val':[0,0,0],
    'maxval':[0xFF,0xFF,0xFF],
    'frequency':[1,0,0],
    'amplitude':[1,0,0]
}

def spiral():

    windowName = 'spiral'

    # read parameters, calculae movment (starting color value), fill in color
    # starting vlue per channel
    stVals = [parameters['val'][i] for i in range(3)]
    # maximum value per chanel
    maxchannels = [parameters['maxval'][i] for i in range(3)]
    # phase angle
    phases = [parameters['phase angle'][i] for i in range(3)]
    stVals = [stVals[i] + (phases[i] * maxchannels[i] / 360) for i in range(3)]
    # color diferential per angle change
    fneDifs = [parameters['fine angle dif'][i] for i in range(3)]
    cseDifs = [parameters['corse angle dif'][i] for i in range(3)]

    # storage for prev1ious positions
    prev1 = None
    prev2 = None

    # width of the picture
    imageSize = parameters['imageSize'] + 1

    # center of image
    origin = int(imageSize / 2)

    # calculations to keep the plot within the immage
    lineThickness = parameters['lineThickness'] + 1
    # calculate number of turn such that the radious of the graph is 1/3 of the screen width
    numTurns = int(imageSize / lineThickness / 3)
    # radius for edge positions
    edge = lineThickness * numTurns
    image = np.full((imageSize,imageSize,3),[0,0,0],dtype=np.uint8)

    # amount to incriment each fill cycle
    inc = parameters['inc'] + 1

    if parameters['graph']['fill']:

        full = 2 * pi

        for i in range(0,360 * numTurns,inc):
            # current radius for fill
            r = i * lineThickness / 360
            # angle of fill in radians
            angle = i * full / 360
            # trig calculations
            trig =  cos(angle),sin(angle)
            # polar to cartisian conversion
            pos1 = [int(origin + trig[j] * r   ) for j in range(2)]
            pos2 = [int(origin + trig[j] * edge) for j in range(2)]
            # calculate color
            color = [(stVals[j] + i * (cseDifs[j] * 180  + fneDifs[j] - 180) / maxchannels[j]) % maxchannels[j] for j in range(3)]
            # filling the position
            if prev1 != None:
                if parameters['type'] == 'line':
                    cv2.line(image,prev1,pos1,color,lineThickness)
                    cv2.line(image,prev2,pos2,color,lineThickness)
                elif parameters['type'] == 'rectangle center':
                    cv2.rectangle(image,prev1,pos1,color,lineThickness)
                elif parameters['type'] == 'rectangle edge':
                    cv2.rectangle(image,prev2,pos2,color,lineThickness)
                elif parameters['type'] == 'circle':
                    cv2.circle(image,pos1,int(lineThickness/2),color,-1)
                    cv2.circle(image,pos2,int(lineThickness/2),color,-1)
                elif parameters['type'] == 'ellipse':
                    MA = parameters['majr axis'] + 1
                    SA = parameters['semi axis'] + 1
                    rotation = parameters['angle']
                    distance = int(sqrt(abs(pos1[0] - prev1[0])**2 + abs(pos1[1] - prev1[1])**2))
                    cv2.ellipse(image, (pos1, (distance/MA,distance/SA),i + rotation), color, -1)
                elif parameters['type'] == 'polygon':
                    # radius of each loop end
                    radii = [((i + ((j > 1) * inc)) * lineThickness / 360) - ([0,1,1,0][j] * lineThickness) for j in range(4)]
                    angles = [(i + (j       * inc)) * full          / 360                                   for j in range(2)]
                    Trigs = [[cos(angles[j]),sin(angles[j])] for j in range(2)]
                    # create an array of points, in a circular fasion from one radii then the next angle and back in the prev1ious radii
                    poly1 = array([[int(origin + Trigs[k > 1][l] * radii[k])                              for l in range (2)] for k in range(4)])
                    poly2 = array([[int(origin + Trigs[k > 1][l] * (edge - [0,1,1,0][k] * lineThickness)) for l in range (2)] for k in range(4)])
                    # convert to degrees then do mod 360 and convert back
                    image = cv2.fillPoly(image,[poly2],color)
                    image = cv2.fillPoly(image,[poly1],color)
                    # for j in range(4):
                    #     cv2.line(image,poly1[j],poly1[(j + 1) % 4],[0xFF,0xFF,0xFF])
                    #     cv2.line(image,poly2[j],poly2[(j + 1) % 4],[0xFF,0xFF,0xFF])

            prev1 = pos1
            prev2 = pos2

    # spiral graph
    if parameters['graph']['spiral']:
        full = 2 * pi
        for i in range(0,360 * numTurns,inc):
            if i > 0:
                radii  = [(i + (j * inc)) * lineThickness / 360 for j in range(2)]
                angles = [(i + (j * inc)) * full          / 360 for j in range(2)]
                carts  = [[int(origin + cos(angles[j]) * radii[j]), int(origin + sin(angles[j]) * radii[j])] for j in range(2)]
                channels = [stVals[j] + i * (cseDifs[j] * 180  + fneDifs[j] - 180) / maxchannels[j] for j in range(3)]
                part = int(channels[0] / maxchannels[0]) % 2
                if part:
                    cv2.line(image,carts[0],carts[1],[0,0xFF,0])
                else:
                    cv2.line(image,carts[0],carts[1],[0,0,0xFF])

    # time cycle graph
    if parameters['graph']['time']:
        full = 2 * pi
        # position
        offset = imageSize / 3
        center = int((imageSize / 2) + offset),int((imageSize / 2) - offset)
        # size
        radius = edge / 2.6
        # width of each loop
        width = radius / 6
        # radius of each loop end
        radii = [(radius - i * width) for i in range(5)]
        chanangles = [(stVals[i] / maxchannels[i]) * full for i in range(3)]
        chanangles.append(parameters['timeAngle'])
        # circular gradiants
        step = int(360 / 60)
        for j in range(0,360,step):
            angles = [(j + k * step) * full / 360 for k in range(2)]
            colorTrigs = [[cos(angles[k]),sin(angles[k])] for k in range(2)]
            for i in range(4):
                # create an array of points, in a circular fasion from one radii then the next angle and back in the prev1ious radii
                poly1 = array([[int(center[l] + colorTrigs[k > 1][l] * radii[i+[0,1,1,0][k]]) for l in range (2)] for k in range(4)])
                # convert to degrees then do mod 360 and convert back
                chanangles[i] = (chanangles[i] * 360 / full % 360) * full / 360
                if chanangles[i] >= angles[0] and chanangles[i] < angles[1]:
                    image = cv2.fillPoly(image,[poly1],[0xFF,0xFF,0xFF])
                else:
                    color = [0,0,0]
                    if i < 3:
                        color[i] = j / 360 * maxchannels[i]
                        image = cv2.fillPoly(image,[poly1],color)
                        cv2.line(image,poly1[0],poly1[3],[0xFF,0xFF,0xFF])
                    cv2.line(image,poly1[1],poly1[2],[0xFF,0xFF,0xFF])

    # phase diagram
    if parameters['graph']['phase']:
        for i in range(0,360 * numTurns,inc):
            x = origin - edge + int((i % 360) * (2 * edge) / 360)
            y = origin + edge + lineThickness + int(i / 360)
            pos1 = x,y + 20
            channels = [(stVals[j] + i * (cseDifs[j] * 180  + fneDifs[j] - 180) / maxchannels[j]) % maxchannels[j] for j in range(3)]
            cv2.circle(image,pos1,0,channels)
            if i % 360 == 0:
                cv2.circle(image,pos1,0,[0xFF,0xFF,0xFF])
            elif i % 90 == 0:
                cv2.circle(image,pos1,0,[0,0xFF,0xFF])
            elif i % 60 == 0:
                cv2.circle(image,pos1,0,[0xFF,0,0xFF])
            elif i % 30 == 0:
                cv2.circle(image,pos1,0,[0xFF,0xFF,0])
            elif i % 15 == 0:
                cv2.circle(image,pos1,0,[0,0,0xFF])

    # unit circle
    if parameters['graph']['unit']:
        cv2.circle(image,(origin,origin),edge,[0xFF,0xFF,0xFF])
        for i in range(6):
            angle = i * (pi/6)
            trigs = int(edge * cos(angle)),int(edge * sin(angle))
            end1 = [origin + trigs[j] for j in range(2)]
            end2 = [origin - trigs[j] for j in range(2)]
            cv2.line(image,end1,end2,[0xFF,0,0xFF])
        for i in range(1,4,2):
            angle = i * (pi/4)
            trigs = int(edge * cos(angle)),int(edge * sin(angle))
            end1 = [origin + trigs[j] for j in range(2)]
            end2 = [origin - trigs[j] for j in range(2)]
            cv2.line(image,end1,end2,[0xFF,0xFF,0])

    # display data as in different colorspace
    if parameters['colorspace'] == 'HSV':
        image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
    if parameters['colorspace'] == 'LUV':
        image = cv2.cvtColor(image,cv2.COLOR_LUV2BGR)
    if parameters['colorspace'] == 'YUV':
        image = cv2.cvtColor(image,cv2.COLOR_YUV2BGR)
    if parameters['colorspace'] == 'LAB':
        image = cv2.cvtColor(image,cv2.COLOR_LAB2BGR)

    cv2.imshow(windowName,image)
    cv2.waitKey(1)

def change(*args):
    print('change',args[1])
    if len(args[1]) > 2:
        parameters[args[1][0]][args[1][1]] = args[1][2]
    else:
        parameters[args[1][0]] = args[1][1]
    if parameters['colorspace'] == 'HSV':
        parameters['maxblu'] = 180
        parameters['maxgen'] = 256
        parameters['maxred'] = 256
    elif parameters['colorspace'] == 'BGR':
        parameters['maxblu'] = 256
        parameters['maxgen'] = 256
        parameters['maxred'] = 256
    pass

# hek

def check(*args):
    print (args)
    print(args[1][0],args[0])
    parameters[args[1][0]][args[1][1]] = args[0]
    pass

def reset(*args):
    # todo: make a reset function that works
    pass

# trackbar callback, one callback to rule them all
def tbcb(key,value):
    print(key,value)
    if type(key) == list:
        parameters[key[0]][key[1]] = value
    else:
        parameters[key] = value

# create track bar
def ctb(tname,wname,dval,mval,callback):
    print('{:25s} {:10s} {:<9d} {:<9d} {}'.format(tname,wname,dval,mval,callback))
    cv2.createTrackbar(tname,wname,dval,mval,callback)

def createSliderWindow():

    windowName = 'controls'

    cv2.namedWindow(windowName)

    height = 120
    width = height * 8
    size = int(height / 40)
    y = int(height * 2 / 3)

    blank = np.empty((height,width,3),dtype=np.uint8)

    for i in range(blank.shape[0]):
        for j in range(blank.shape[1]):
            for k in range(3):
                blank[i][j][k] *= 2/3

    blank = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)

    cv2.putText(blank,windowName,(30,y),cv2.FORMATTER_FMT_DEFAULT,size,(100, 0, 0),3)
    cv2.putText(blank,windowName,(30,y),cv2.FORMATTER_FMT_DEFAULT,size,(200, 0, 0),2)
    cv2.imshow(windowName,blank)

    ctb('imageSize'          ,windowName,parameters['imageSize']    ,1000,lambda v:tbcb('imageSize'    ,v))
    ctb('clock'              ,windowName,parameters['clock']        ,1000,lambda v:tbcb('clock'        ,v))
    ctb('incriment'          ,windowName,parameters['inc']          ,359 ,lambda v:tbcb('inc'          ,v))
    ctb('lineThickness'      ,windowName,parameters['lineThickness'],256 ,lambda v:tbcb('lineThickness',v))
    ctb('majr axis (ellipse)',windowName,parameters['majr axis']    ,30  ,lambda v:tbcb('majr axis'    ,v))
    ctb('semi axis (ellipse)',windowName,parameters['semi axis']    ,30  ,lambda v:tbcb('semi axis'    ,v))
    ctb('angle     (ellipse)',windowName,parameters['angle']        ,360 ,lambda v:tbcb('angle'        ,v))

    suffix = ['blue','green','red']

    widecfgs = ['speed','angle dif']

    trigs = ['sin','cos','tan','none']

    cfgs = ['frequency','amplitude','phase angle']

    buttonTypes = ['direction','constant']

    for i in range(len(suffix)):
        a = int(i)
        color = suffix[a]
        for r in range(len(widecfgs)):
            wcfg = widecfgs[r]
            ctb(color + ' corse ' + wcfg,windowName,parameters['corse ' + wcfg][a],360,lambda v:tbcb(['corse ' + wcfg,a],v))
            ctb(color + ' fine '  + wcfg,windowName,parameters['fine '  + wcfg][a],360,lambda v:tbcb(['fine '  + wcfg,a],v))

        for b in range(len(cfgs)):
            cfg = cfgs[b]
            ctb(color + ' ' + cfg  ,windowName,parameters[cfg][a],360 ,lambda v:tbcb([cfg,a],v))

        for f in range(len(buttonTypes)):
            btype = buttonTypes[f]
            cv2.createButton(color + ' ' + btype,check,[btype,a],1,0)

        for u in range(len(trigs)):
            trig = trigs[u]
            cv2.createButton(color + ' ' + trig,change,['trig',a,trig] ,2,0)

    types = ['rectangle center','rectangle edge','line','circle','ellipse','polygon']

    for k in range(len(types)):
        type = types[k]
        cv2.createButton(type + ' graph',change,['type',type],2,parameters['type'] == type)

    graphTypes =  ['fill','phase','spiral','unit','time']

    for j in range(len(graphTypes)):
        type = graphTypes[j]
        cv2.createButton(type + ' graph',check,['graph',type],1,parameters['graph'][type])

    colorspaces = ['BGR','HSV','LUV','YUV','LAB']

    for l in range(len(colorspaces)):
        cv2.createButton(colorspaces[l] + ' colorspace',change,['colorspace',colorspaces[l]],2,parameters['colorspace'] == types[k])

    cv2.createButton('reset',reset)


def calculateMovement(oldTime):
    newTime = time_ns()
    clock = parameters['clock'] + 1
    deltaTime = (newTime - oldTime) / (clock / 60)
    deltatimeangle = deltaTime / 1e+9 * pi * 2
    timeAngle = parameters['timeAngle']
    full = 2 * pi
    if timeAngle != None:
        timeAngle += deltatimeangle
    else:
        timeAngle = deltatimeangle

    parameters['timeAngle'] = timeAngle

    maxchannels = [parameters['maxval'][i] for i in range(3)]

    # load start values
    stVals = [parameters['val'][i] for i in range(3)]

    # trig
    for i in range(3):
        trig = parameters['trig'][i]
        if trig != 'none':
            frequencyResponce = parameters['frequency'][i] / 60
            amplitudeResponce = parameters['amplitude'][i]
        if trig == 'sin':
            stVals[i] = sin(timeAngle * frequencyResponce) * amplitudeResponce
        elif trig == 'cos':
            stVals[i] = cos(timeAngle * frequencyResponce) * amplitudeResponce
        elif trig == 'tan':
            stVals[i] = tan(timeAngle * frequencyResponce) * amplitudeResponce

    # constant
    for i in range(3):
        if parameters['constant'][i]:
            fnespeed = parameters['fine speed'][i]
            csespeed = parameters['corse speed'][i]
            if parameters['direction'][i]:
                stVals[i] += (csespeed + (fnespeed / maxchannels[i])) * (deltaTime / 1e+8)
            else:
                stVals[i] -= (csespeed - (fnespeed / maxchannels[i])) * (deltaTime / 1e+8)
        parameters['val'][i] = stVals[i]

    return newTime

if __name__ == '__main__':
    createSliderWindow()
    time = time_ns()
    while True:
        time = calculateMovement(time)
        spiral()
