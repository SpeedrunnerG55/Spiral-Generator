import cv2
import imageio
import numpy as np
from numpy import array
from math import sin, cos, tan, pi, sqrt
from time import time_ns

# defaults
parameters = {
    'save':False,
    'immageWidth':500,
    'graph':{'fill':True,'phase':True,'spiral':False,'unit':False,'time':True,'colorspace':True},
    'colorspace':'BGR',
    'clock':100,
    'timeAngle':None,
    'lineWidth':10,
    'incriment':70,
    'fill':'polygon',
    'majr axis':0,
    'semi axis':0,
    'angle':0,
    'phase angle':[0,0,0],
    'corse angle dif':[1,1,1],
    'fine angle dif':[179,179,179],
    'trig':['none','none','none'],
    'direction':[False,False,False],
    'fine speed':[0,0,0],
    'corse speed':[20,10,4],
    'constant':[True,True,True],
    'val':array([0,0,0],dtype=np.uint8),
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

    # calculate start value
    stVals = [stVals[i] + (phases[i] * maxchannels[i] / 360) for i in range(3)]

    # calculations to keep the plot within the immage
    lineWidth = parameters['lineWidth'] + 1

    # width of the picture
    immageWidth = parameters['immageWidth'] + 1
    immageHeight = int(immageWidth * 1.3)

    # center of image
    origin = int(immageWidth / 2)

    # calculate number of turn such that the radious of the graph is 1/3 of the screen width
    fillRadius = int(immageWidth / 3)

    # maximum number of turns for a giver linethickness can fit in the fill raduis
    numTurns = int(fillRadius / lineWidth)

    # max value for any fill radius
    maxR = (numTurns - 1) * lineWidth

    # storage for image
    image = np.full((immageHeight,immageWidth,3),[0,0,0],dtype=np.uint8)

    # amount to incriment each fill cycle
    incriment = parameters['incriment'] + 1

    # color diferential per angle change
    fneDifs = [parameters['fine angle dif'][i] for i in range(3)]
    cseDifs = [parameters['corse angle dif'][i] for i in range(3)]
    chandiffs = [(cseDifs[j] * 180  + fneDifs[j] - 180) / maxchannels[j] for j in range(3)]

    MA = parameters['majr axis'] + 1
    SA = parameters['semi axis'] + 1
    rotation = parameters['angle']

    graphlist = parameters['graph']

    widthratio = lineWidth / 360

    graphOffset = int(immageWidth / 20)

    incrimentratio = 360/incriment

    radiansRatio = 2 * pi / 360

    if graphlist['fill']:
        # storage for previous positions
        prev = None
        full = 2 * pi
        fillType = parameters['fill']
        if fillType == 'polygon':
            points = [None,None]
        for i in range(0,(360 * numTurns)+1,incriment):
            # current fillRadius for fill
            r = (i * widthratio)
            # limit radius to edge of fill graph
            if r > maxR:
                r = maxR
            # angle of fill in radians
            angle = i * full / 360
            # trig calculations
            trig = [cos(angle),sin(angle)]
            # calculate color
            color = [(stVals[j] + i * chandiffs[j]) % maxchannels[j] for j in range(3)]
            # filling the graph
            if fillType == 'polygon':
                # Radius, and one inwards
                radii = [r,(i * widthratio) - lineWidth]
                # no negitive radius shenanagans
                if radii[1] < 0:
                    radii[1] = 0
                # get points for these two radii
                points[0] = [[int(origin + trig[l] * radii[k]) for l in range (2)] for k in range(2)]
                if i > 0:
                    # join the last to points in reverse order with the currect two points into an array
                    poly = array(points[0] + [points[1][1],points[1][0]])
                    # apply polygon
                    image = cv2.fillPoly(image,[poly],color)
                # save the points
                points[1] = points[0]
            else:
                # polar to cartisian conversion
                pos = [int(origin + trig[j] * r) for j in range(2)]
                if i > 0:
                    if fillType == 'line':
                        cv2.line(image,prev,pos,color,lineWidth)
                    elif fillType == 'rectangle center':
                        cv2.rectangle(image,prev,pos,color,lineWidth)
                    elif fillType == 'circle':
                        cv2.circle(image,pos,int(lineWidth/2),color,-1)
                    elif fillType == 'ellipse':
                        distance = int(sqrt(abs(pos[0] - prev[0])**2 + abs(pos[1] - prev[1])**2))
                        cv2.ellipse(image, (pos, (distance/MA,distance/SA),i + rotation), color, -1)
                prev = pos

    # spiral graph
    if graphlist['spiral']:
        for i in range(0,360 * numTurns,incriment):
            if i > 0:
                angles = [i * radiansRatio, (i - incriment) * radiansRatio]
                radii  = [i * widthratio,   (i - incriment) * widthratio]
                for i in range(2):
                    if radii[i] > maxR:
                        radii[i] = maxR
                carts  = [[int(origin + cos(angles[j]) * radii[j]), int(origin + sin(angles[j]) * radii[j])] for j in range(2)]
                channels = array([stVals[j] + i * chandiffs[j] for j in range(3)],dtype=np.uint8)
                part = int(channels[0] / maxchannels[0]) % 2
                if part:
                    cv2.line(image,carts[0],carts[1],[0,0xFF,0])
                else:
                    cv2.line(image,carts[0],carts[1],[0,0,0xFF])

    # time cycle graph
    if graphlist['time']:
        full = 2 * pi
        # size
        timeRadius = fillRadius / 3
        # position
        center = int((immageWidth) - timeRadius - graphOffset),int(timeRadius + graphOffset)
        # width of each loop
        width = timeRadius / 6
        # fillRadius of each loop end
        radii = [(timeRadius - i * width) for i in range(5)]
        chanangles = [(stVals[i] / maxchannels[i]) * full for i in range(3)]
        chanangles.append(parameters['timeAngle'])
        # circular gradiants
        step = int(360 / 60)
        for j in range(0,360,step):
            angles = [(j + k * step) * full / 360 for k in range(2)]
            colorTrigs = [[cos(angles[k]),sin(angles[k])] for k in range(2)]
            for i in range(4):
                # create an array of points, in a circular fasion from one radii then the next angle and back in the previous radii
                poly = array([[int(center[l] + colorTrigs[k > 1][l] * radii[i+[0,1,1,0][k]]) for l in range (2)] for k in range(4)])
                # convert to degrees then do mod 360 and convert back
                chanangles[i] = (chanangles[i] * 360 / full % 360) * full / 360
                if chanangles[i] >= angles[0] and chanangles[i] < angles[1]:
                    image = cv2.fillPoly(image,[poly],[0xFF,0xFF,0xFF])
                else:
                    if parameters['colorspace'] == 'HSV':
                        color = [stVals[0],150,200]
                    else:
                        color = [0,0,0]
                    if i < 3:
                        color[i] = j / 360 * maxchannels[i]
                        image = cv2.fillPoly(image,[poly],color)
                        if width > 3:
                            cv2.line(image,poly[0],poly[3],[0xFF,0xFF,0xFF])
                    if width > 3:
                        cv2.line(image,poly[1],poly[2],[0xFF,0xFF,0xFF])

    # colorspace graph
    if graphlist['colorspace']:
        full = 2 * pi
        # position
        width = int(immageWidth / 4)
        inc = int(sqrt(width))
        if inc == 0:
            inc = 1
        width = inc**2
        if width > int(immageWidth / 3):
            width = int(immageWidth / 3)
        tl = graphOffset
        br = graphOffset + width - 1
        if width > 8:
            cv2.rectangle(image,(tl - 1,tl - 1),(br + 1,br + 1),[255,255,255],1)
        for i in range(0,width,inc):
            blu = i / (width) * maxchannels[0]
            for j in range(0,width,inc):
                grn = j / (width) * maxchannels[1]
                for k in range(inc):
                    for l in range(inc):
                        red = k / (inc) * maxchannels[2]
                        color = array([blu,grn,red],dtype=np.uint8)
                        y = (i + graphOffset + k) % immageWidth
                        x = (j + graphOffset + l) % immageWidth
                        distance = sqrt(abs(stVals[0]-color[0])**2 + abs(stVals[1]-color[1])**2 + abs(stVals[2]-color[2])**2)
                        if distance < 60:
                            image[y][x] = color

    # phase diagram
    if graphlist['phase']:
        left = origin - 180
        right = origin + 180
        if left < graphOffset:
            left = graphOffset
            right = immageWidth - graphOffset
        width = right - left
        bottom = immageHeight - graphOffset
        top = bottom - numTurns
        inc = int(incriment * width / 360)
        if inc == 0:
            inc = 1
        if width < 360 and inc > 1:
            width = int(incrimentratio * inc)
            left = origin - int(width/2)
            right = left + width
        tl = (left - 1, top - 1)
        br = (right, bottom)
        if numTurns > 0:
            cv2.rectangle(image,tl,br,[255,255,255],1)
        ratio = 360 / width
        end = width * numTurns
        colorspace = parameters['colorspace']
        for i in range(0,end,inc):
            deg = int(i * ratio)
            channels = [(stVals[j] + deg * chandiffs[j]) % maxchannels[j] for j in range(3)]
            x = left + int(i % width)
            y = top + int(i / width)
            # image[y][x] = channels
            if deg % 180 == 0:
                image[y][x] = [0x80,0x80,0x80]
            elif deg % 90 == 0:
                image[y][x] = [0x80,0,0x80]
            elif deg % 45 == 0:
                image[y][x] = [0x80,0x80,0]
            elif deg % 30 == 0:
                image[y][x] = [0,0x80,0x80]
            for j in range(3):
                if channels[j] > maxchannels[j] - 30:
                    if colorspace == 'HSV' and k == 0:
                        image[y][x][j] = channels[j]
                    elif colorspace == 'HSV' and k == 2:
                        image[y][x][j] = 255
                    else:
                        image[y][x][j] = 255
                    for k in range(3):
                        if image[y][x][k] == 0x80:
                            image[y][x][k] = 0



    # unit circle
    if graphlist['unit']:
        cv2.circle(image,(origin,origin),fillRadius,[0xFF,0xFF,0xFF])
        # importiant angles
        pioversix = pi/6
        pioverfour = pi/4
        for i in range(6):
            angle = i * pioversix
            trigs = int(fillRadius * cos(angle)),int(fillRadius * sin(angle))
            end1 = [origin + trigs[j] for j in range(2)]
            end2 = [origin - trigs[j] for j in range(2)]
            cv2.line(image,end1,end2,[0xFF,0,0xFF])
        for i in range(1,4,2):
            angle = i * pioverfour
            trigs = int(fillRadius * cos(angle)),int(fillRadius * sin(angle))
            end1 = [origin + trigs[j] for j in range(2)]
            end2 = [origin - trigs[j] for j in range(2)]
            cv2.line(image,end1,end2,[0xFF,0xFF,0])

    # unload parameter
    colorspace = parameters['colorspace']

    # display data as in different colorspace
    if colorspace != 'BGR':
        if colorspace == 'HSV':
            code = cv2.COLOR_HSV2BGR
        if colorspace == 'LUV':
            code = cv2.COLOR_LUV2BGR
        if colorspace == 'YUV':
            code = cv2.COLOR_YUV2BGR
        if colorspace == 'LAB':
            code = cv2.COLOR_LAB2BGR
        image = cv2.cvtColor(image,code)

    cv2.imshow(windowName,image)
    cv2.waitKey(1)
    return image

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

def createControlls():

    windowName = 'controls'

    cv2.namedWindow(windowName)

    height = 120
    width = height * 8
    size1 = int(height / 40)
    size2 = int(height / 120)
    y1 = int(height * 3 / 5)
    y2 = int(height * 7 / 8)

    blank = np.empty((height,width,3),dtype=np.uint8)

    for i in range(blank.shape[0]):
        for j in range(blank.shape[1]):
            for k in range(3):
                blank[i][j][k] += j / blank.shape[1] * 200
                blank[i][j][k] *= 3/5


    blank = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)

    cv2.putText(blank,windowName,(30,y1),cv2.FORMATTER_FMT_DEFAULT,size1,(100, 0, 0),3)
    cv2.putText(blank,windowName,(30,y1),cv2.FORMATTER_FMT_DEFAULT,size1,(200, 0, 0),2)

    cv2.putText(blank,'ctrl + p for more controls',(30,y2),cv2.FORMATTER_FMT_DEFAULT,size2,(100, 0, 0),3)
    cv2.putText(blank,'ctrl + p for more controls',(30,y2),cv2.FORMATTER_FMT_DEFAULT,size2,(200, 0, 0),2)

    cv2.imshow(windowName,blank)

    suffix = ['blue','green','red']

    general_configs = {
        'immageWidth':1000,
        'clock':1000,
        'incriment':359,
        'lineWidth':256,
        'majr axis':30,
        'semi axis':30,
        'angle':360
   }

    for cfg in general_configs:
        ctb(cfg,windowName,parameters[cfg],general_configs[cfg],lambda v, cfg=cfg :tbcb(cfg,v))

    channel_cfgs = {'frequency':60,'amplitude':256,'phase angle':360}

    widecfgs = {'speed':[60,60],'angle dif':[30,360]}

    buttonTypes = ['direction','constant']

    trigs = ['sin','cos','tan','none']

    for color in suffix:
        a = suffix.index(color)
        for wcfg in widecfgs:
            ctb(color + ' corse ' + wcfg,windowName,parameters['corse ' + wcfg][a],widecfgs[wcfg][0],lambda v, a=a, wcfg=wcfg:tbcb(['corse ' + wcfg,a],v))
            ctb(color + ' fine '  + wcfg,windowName,parameters['fine '  + wcfg][a],widecfgs[wcfg][1],lambda v, a=a, wcfg=wcfg:tbcb(['fine '  + wcfg,a],v))

        for cfg in channel_cfgs:
            ctb(color + ' ' + cfg  ,windowName,parameters[cfg][a],channel_cfgs[cfg],lambda v, a=a, cfg=cfg:tbcb([cfg,a],v))

        for f in range(len(buttonTypes)):
            btype = buttonTypes[f]
            cv2.createButton(color + ' ' + btype,check,[btype,a],1,parameters[btype][a])

        for u in range(len(trigs)):
            trig = trigs[u]
            cv2.createButton(color + ' ' + trig,change,['trig',a,trig] ,2,parameters['trig'][a] == trig)

    filltypes = ['rectangle center','line','circle','ellipse','polygon']

    for fill in filltypes:
        cv2.createButton(fill + ' fill',change,['fill',fill],2,parameters['fill'] == fill)

    graphTypes =  ['fill','phase','spiral','unit','time','colorspace']

    for graph in graphTypes:
        cv2.createButton(graph + ' graph',check,['graph',graph],1,parameters['graph'][graph])

    colorspaces = ['BGR','HSV','LUV','YUV','LAB']

    for code in colorspaces:
        cv2.createButton(code + ' colorspace',change,['colorspace',code],2,parameters['colorspace'] == code)

    cv2.createButton('reset',reset)

    cv2.createButton('save',save)


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

def save(*args):
    # lol why did i even make this a funciton, or right i had to DX
    parameters['save'] = True

if __name__ == '__main__':
    createControlls()
    time = time_ns()
    first = None

    frames = []
    image_count = 0

    while True:
        time = calculateMovement(time)
        frame = spiral()
        if parameters['save']:
            equal = np.array_equal(first,frame)
            if equal or image_count > 500:
                break
            else:
                frames.append(frame)
                image_count += 1

            if np.array_equal(first,None):
                first = frame

    print("Saving GIF file")
    with imageio.get_writer("spiral.gif", mode="I") as writer:
        for idx, frame in enumerate(frames):
            print("Adding frame to GIF file: ", idx + 1)
            writer.append_data(frame)
