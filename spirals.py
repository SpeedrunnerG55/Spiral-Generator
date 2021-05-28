import cv2
import numpy as np
from numpy import array
from math import sin, cos, tan, pi, sqrt
from time import time_ns

# defaults
parameters = {
    'imageSize':500,
    'graph':{'fill':True,'phase':False,'spiral':False,'unit':False,'time':True},
    'colorspace':'BGR',
    'clock':100,
    'timeAngle':None,
    'lineThickness':20,
    'incriment':0,
    'fill':'polygon',
    'majr axis':0,
    'semi axis':0,
    'angle':0,
    'phase angle':[0,0,0],
    'corse angle dif':[4,0,0],
    'fine angle dif':[179,179,179],
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
    incriment = parameters['incriment'] + 1

    if parameters['graph']['fill']:

        full = 2 * pi

        for i in range(0,360 * numTurns,incriment):
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
                if parameters['fill'] == 'line':
                    cv2.line(image,prev1,pos1,color,lineThickness)
                    cv2.line(image,prev2,pos2,color,lineThickness)
                elif parameters['fill'] == 'rectangle center':
                    cv2.rectangle(image,prev1,pos1,color,lineThickness)
                elif parameters['fill'] == 'rectangle edge':
                    cv2.rectangle(image,prev2,pos2,color,lineThickness)
                elif parameters['fill'] == 'circle':
                    cv2.circle(image,pos1,int(lineThickness/2),color,-1)
                    cv2.circle(image,pos2,int(lineThickness/2),color,-1)
                elif parameters['fill'] == 'ellipse':
                    MA = parameters['majr axis'] + 1
                    SA = parameters['semi axis'] + 1
                    rotation = parameters['angle']
                    distance = int(sqrt(abs(pos1[0] - prev1[0])**2 + abs(pos1[1] - prev1[1])**2))
                    cv2.ellipse(image, (pos1, (distance/MA,distance/SA),i + rotation), color, -1)
                elif parameters['fill'] == 'polygon':
                    # radius of each loop end
                    radii = [((i + ((j > 1) * incriment)) * lineThickness / 360) - ([0,1,1,0][j] * lineThickness) for j in range(4)]
                    angles = [(i + (j       * incriment)) * full          / 360                                   for j in range(2)]
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
        for i in range(0,360 * numTurns,incriment):
            if i > 0:
                radii  = [(i + (j * incriment)) * lineThickness / 360 for j in range(2)]
                angles = [(i + (j * incriment)) * full          / 360 for j in range(2)]
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
                    if parameters['colorspace'] == 'HSV':
                        color = [stVals[0],150,200]
                    else:
                        color = [0,0,0]
                    if i < 3:
                        color[i] = j / 360 * maxchannels[i]
                        image = cv2.fillPoly(image,[poly1],color)
                        cv2.line(image,poly1[0],poly1[3],[maxchannels[0],maxchannels[1],maxchannels[2]])
                    cv2.line(image,poly1[1],poly1[2],[0xFF,0xFF,0xFF])

    # phase diagram
    if parameters['graph']['phase']:
        for i in range(0,360 * numTurns,incriment):
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
        parameters['maxval'][0] = 180
        parameters['maxval'][1] = 256
        parameters['maxval'][2] = 256
    elif parameters['colorspace'] == 'BGR':
        parameters['maxval'][0] = 256
        parameters['maxval'][1] = 256
        parameters['maxval'][2] = 256
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
    # print('{:25s} {:10s} {:<9d} {:<9d} {}'.format(tname,wname,dval,mval,callback))
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

    suffix = ['blue','green','red']

    trigs = ['sin','cos','tan','none']

    buttonTypes = ['direction','constant']

    general_configs = {
        'imageSize':1000,
        'clock':1000,
        'incriment':359,
        'lineThickness':256,
        'majr axis':30,
        'semi axis':30,
        'angle':360
   }

    for cfg in general_configs:
        ctb(cfg,windowName,parameters[cfg],general_configs[cfg],lambda v, cfg=cfg :tbcb(cfg,v))

    channel_cfgs = {'frequency':60,'amplitude':256,'phase angle':360}

    widecfgs = {'speed':[60,60],'angle dif':[30,360]}

    for color in suffix:
        a = suffix.index(color)
        for wcfg in widecfgs:
            ctb(color + ' corse ' + wcfg,windowName,parameters['corse ' + wcfg][a],widecfgs[wcfg][0],lambda v, a=a, wcfg=wcfg:tbcb(['corse ' + wcfg,a],v))
            ctb(color + ' fine '  + wcfg,windowName,parameters['fine '  + wcfg][a],widecfgs[wcfg][1],lambda v, a=a, wcfg=wcfg:tbcb(['fine '  + wcfg,a],v))

        for cfg in channel_cfgs:
            ctb(color + ' ' + cfg  ,windowName,parameters[cfg][a],channel_cfgs[cfg],lambda v, a=a, cfg=cfg:tbcb([cfg,a],v))

        for f in range(len(buttonTypes)):
            btype = buttonTypes[f]
            cv2.createButton(color + ' ' + btype,check,[btype,a],1,0)

        for u in range(len(trigs)):
            trig = trigs[u]
            cv2.createButton(color + ' ' + trig,change,['trig',a,trig] ,2,0)

    filltypes = ['rectangle center','rectangle edge','line','circle','ellipse','polygon']

    for fill in filltypes:
        cv2.createButton(fill + ' fill',change,['fill',fill],2,parameters['fill'] == fill)

    graphTypes =  ['fill','phase','spiral','unit','time']

    for graph in graphTypes:
        cv2.createButton(graph + ' graph',check,['graph',graph],1,parameters['graph'][graph])

    colorspaces = ['BGR','HSV','LUV','YUV','LAB']

    for code in colorspaces:
        cv2.createButton(code + ' colorspace',change,['colorspace',code],2,parameters['colorspace'] == code)

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
                stVals[i] += (csespeed + (fnespeed / 60)) * (deltaTime / 1e+8)
            else:
                stVals[i] -= (csespeed - (fnespeed / 60)) * (deltaTime / 1e+8)
        parameters['val'][i] = stVals[i]

    return newTime

if __name__ == '__main__':
    createSliderWindow()
    time = time_ns()
    while True:
        time = calculateMovement(time)
        spiral()
