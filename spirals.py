import cv2
import imageio
import numpy as np
from numpy import array
from math import sin, cos, tan, pi, sqrt, ceil, log
from time import time_ns, sleep
import random

flags = {
    'test':False,
    'save':False,
    'timing':False
}

# defaults
parameters = {
    'maxSize':600,
    'uiColor':[0xDD,0xDD,0xDD],
    'image width':500,
    'graph':{'fill':True,'phase':True,'spiral':False,'unit':False,'time':True,'colorspace':True},
    'colorspace':'BGR',
    'clockInterval':100,
    'timeAngle':0,
    'lineWidth':10,
    'incriment':70,
    'fill':'polygon',
    'majr axis':0,
    'semi axis':0,
    'angle':0,
    'phase threshold':25,
    'colorspace threshold':25,
    'phase angle':[0,0,0],
    'corse angle dif':[1,1,1],
    'fine angle dif':[179,179,179],
    'trig':['none','none','none'],
    'direction':[False,False,False],
    'fine speed':[0,0,0],
    'corse speed':[20,10,4],
    'constant':[True,True,True],
    'val':[0,0,0],
    'maxval':[0xFF,0xFF,0xFF],
    'frequency':[1,0,0],
    'amplitude':[1,0,0]
}

time = 0
windowName = 'spiral'
fullRadians = 2 * pi
radiansRatio = fullRadians / 360
degreesRatio = 360 / fullRadians

def start():
    global time
    time = time_ns()

def stop():
    global time
    delta = time_ns() - time
    time = time_ns()
    return delta

def spiral():


    # read parameters, calculae movment (starting color value), fill in color
    # starting vlue per channel
    stVals = [parameters['val'][i] for i in range(3)]

    # maximum value per chanel
    maxchannels = [parameters['maxval'][i] for i in range(3)]

    # phase angle
    phases = [parameters['phase angle'][i] for i in range(3)]

    # calculate start value
    stVals = [stVals[i] + (phases[i] * maxchannels[i] / 360) for i in range(3)]

    # restrict values
    stVals = [stVals[i] % maxchannels[i] for i in range(3)]

    # calculations to keep the plot within the image
    lineWidth = parameters['lineWidth'] + 1

    timing = flags['timing']

    # width of the picture
    imagewidth = parameters['image width'] + 1

    # storage for image
    image = np.zeros((int(imagewidth * 1.3),imagewidth,3),dtype=np.uint8)

    # center of image
    origin = int(image.shape[1] / 2)

    # calculate number of turn such that the radious of the graph is 1/3 of the screen width
    fillRadius = int(image.shape[1] / 3)

    # maximum number of turns for a giver linethickness can fit in the fill raduis
    numTurns = int(fillRadius / lineWidth)

    # max value for any fill radius
    maxR = (numTurns - 1) * lineWidth

    # amount to incriment each fill cycle
    incriment = parameters['incriment'] + 1

    # color diferential per angle change
    fneDifs = [parameters['fine angle dif'][i] for i in range(3)]
    cseDifs = [parameters['corse angle dif'][i] for i in range(3)]
    chandiffs = [(cseDifs[j] * 180  + fneDifs[j] - 180) / maxchannels[j] for j in range(3)]

    # parameters for ellipsoids
    MA = parameters['majr axis'] + 1
    SA = parameters['semi axis'] + 1
    rotation = parameters['angle']

    # distance from edge for each graph (other than fill)
    graphOffset = int(image.shape[1] / 20)

    # constant


    # conversion factors
    widthratio = lineWidth / 360
    incrimentRatio = 360/incriment


    # define colors
    uiColor = parameters['uiColor']

    # list of what graphs to graph
    graphlist = parameters['graph']

    if timing:
        times = []
        names = []
        start()

    if graphlist['fill']:
        # storage for previous positions
        prev = None
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
            angle = i * radiansRatio
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

        if timing:
            times.append(stop())
            names.append('fill')

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

        if timing:
            times.append(stop())
            names.append('spiral')

    # time cycle graph
    if graphlist['time']:

        # size
        timeRadius = int(image.shape[1] / 10)
        # position
        center = int((image.shape[1]) - timeRadius - graphOffset),int(timeRadius + graphOffset)
        # width of each loop
        width = timeRadius / 6
        # fillRadius of each loop end
        radii = [(timeRadius - i * width) for i in range(5)]
        chanangles = [(stVals[i] / maxchannels[i]) * fullRadians for i in range(3)]
        chanangles.append(parameters['timeAngle'])
        # circular gradiants
        step = 10
        points = [None,None]
        plotted = [False] * 4
        for j in range(0,361,step):
            angleRatio = j / 360
            angle = j * radiansRatio
            trig = [cos(angle),sin(angle)]
            points[0] = [[int(center[l] + trig[l] * radii[k]) for l in range (2)] for k in range(5)]
            if j > 0:
                for i in range(4):
                    # good luck
                    poly = array([points[0][i],points[0][i+1],points[1][i+1],points[1][i]])
                    if angle >= chanangles[i] and not plotted[i]:
                        image = cv2.fillPoly(image,[poly],uiColor)
                        plotted[i] = True
                    else:
                        color = [0,0,0]
                        if i < 3:
                            if parameters['colorspace'] == 'HSV':
                                if i == 0:
                                    color = [angleRatio * maxchannels[0],150,200]
                                elif i == 1:
                                    color = [stVals[0],angleRatio * maxchannels[1],200]
                                elif i == 2:
                                    color = [stVals[0],150,angleRatio * maxchannels[2]]
                            else:
                                color[i] = angleRatio * maxchannels[i]
                        image = cv2.fillPoly(image,[poly],color)
                        if i == 0:
                            cv2.line(image,poly[0],poly[3],uiColor)
                        if i == 3:
                            cv2.line(image,poly[1],poly[2],uiColor)
            points[1] = points[0]

        if timing:
            times.append(stop())
            names.append('time')

    # colorspace graph
    if graphlist['colorspace']:
        threshold = parameters['colorspace threshold']
        # position
        graphwidth = int(image.shape[1] / 5)

        # adjust graph width to always be a perfect square
        inc = int(sqrt(graphwidth))
        if inc == 0:
            inc = 1
        graphwidth = inc**2

        if graphwidth > int(image.shape[1] / 3):
            graphwidth = int(image.shape[1] / 3)
        tl = graphOffset
        br = graphOffset + graphwidth - 1
        cv2.rectangle(image,(tl - 1,tl - 1),(br + 1,br + 1),uiColor,1)
        closestColor = None
        point_count = 0
        point = None
        minDistance = threshold
        div = 3
        for i in range(0,graphwidth,inc):
            blu = i / graphwidth * maxchannels[0]
            for j in range(0,graphwidth,inc):
                grn = j / graphwidth * maxchannels[1]
                x = j + graphOffset
                for k in range(inc):
                    red = k / inc * maxchannels[2]
                    y = i + graphOffset + k
                    color = [blu,grn,red]
                    distance = sqrt(abs(stVals[0]-color[0])**2 + abs(stVals[1]-color[1])**2 + abs(stVals[2]-color[2])**2)
                    if distance < threshold:
                        if distance < minDistance:
                            closestColor = color
                            minDistance = distance
                            point = (x,y)
                        point_count += 1
                    color = [color[0] / div,color[1] / div,color[2] / div]
                    for l in range(inc):
                        image[y][x + l] = color
        # calculate the radius of the circle of points over the threshhold given the number of points (area)
        if timing:
            times.append(stop())
            names.append('colorspace loop')
        circlewidth = int(sqrt(point_count))
        if circlewidth > 0:
            div = 2
            graph = np.zeros(image.shape,dtype=np.uint8)
            cv2.circle(graph,point,circlewidth,uiColor,-1)
            if circlewidth > 1:
                cv2.circle(graph,point,circlewidth - 1,closestColor,-1)
            # masking out the part of the graph i want, why not just make a small graph, i probably would not have to deal with mask hell
            graphmask = np.zeros(image.shape[:2],dtype=np.uint8)
            cv2.rectangle(graphmask,(tl,tl),(br,br),255,-1)
            graph = cv2.bitwise_and(graph,graph,mask=graphmask)
            # mask for the image to remove the portion covered by graph
            imagemask = np.full(image.shape[:2],255,dtype=np.uint8)
            # mark the portion of the graph as 0
            cv2.circle(imagemask,point,circlewidth,0,-1)
            # this is just silly (mask hell)
            # flip the portion of the mask within the original mask from being marked with 0 to marked with 255
            imagemask = cv2.bitwise_not(imagemask,mask=graphmask)
            # now only the graphmask portion of the imagemask is marked 255
            # flip the entire mask so only the graphmask portion of the imagemask is marked 0 aka
            # i want to mask out the graphed regon from image, and only that region
            imagemask = cv2.bitwise_not(imagemask)
            # apply mask
            image = cv2.bitwise_and(image,image,mask=imagemask)
            # now everything is setup for compossiting...
            # composite
            image = image + graph

        if timing:
            times.append(stop())
            names.append('colorspace mask')

    # phase diagram
    if graphlist['phase']:
        div = 3
        threshold = parameters['phase threshold']
        left = origin - 180
        right = origin + 180
        if left < graphOffset:
            left = graphOffset
            right = image.shape[1] - graphOffset
        width = right - left
        bottom = image.shape[0] - graphOffset
        top = bottom - numTurns
        inc = int(incriment * width / 360)
        if inc == 0:
            inc = 1
        if width < 360 and inc > 1:
            width = int(incrimentRatio * inc)
            left = origin - int(width/2)
            right = left + width
        tl = (left - 1, top - 1)
        br = (right, bottom)
        if numTurns > 0 and width > 5:
            cv2.rectangle(image,tl,br,uiColor,1)
        ratio = 360 / width
        end = width * numTurns
        colorspace = parameters['colorspace']
        for i in range(0,end,inc):
            deg = int(i * ratio)
            channels = [(stVals[j] + deg * chandiffs[j]) % maxchannels[j] for j in range(3)]
            x = left + int(i % width)
            y = top + int(i / width)
            image[y][x] = [channels[0] / div,channels[1] / div,channels[2] / div]
            if deg % 180 == 0:
                image[y][x] = [0x80,0x80,0x80]
            elif deg % 90 == 0:
                image[y][x] = [0x80,0,0x80]
            elif deg % 45 == 0:
                image[y][x] = [0x80,0x80,0]
            elif deg % 30 == 0:
                image[y][x] = [0,0x80,0x80]
            for j in range(3):
                if channels[j] > maxchannels[j] - threshold:
                    if colorspace == 'HSV' and j == 0:
                        image[y][x][j] = 179
                    elif colorspace == 'HSV' and j == 2:
                        image[y][x][j] = 255
                    else:
                        image[y][x][j] = 255
                    for k in range(3):
                        if image[y][x][k] == 0x80:
                            image[y][x][k] = 0

        if timing:
            times.append(stop())
            names.append('phase')

    # unit circle
    if graphlist['unit']:
        cv2.circle(image,(origin,origin),fillRadius,uiColor)
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

        if timing:
            times.append(stop())
            names.append('unit')

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

    if timing:
        return times,names,image
    else:
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
        'image width':1000,
        'clockInterval':1000,
        'incriment':359,
        'lineWidth':256,
        'majr axis':30,
        'semi axis':30,
        'angle':360,
        'phase threshold':360,
        'colorspace threshold':255,
        'maxSize':2000
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

    cv2.createButton('run benchmark test',runtest)

def calculateMovement():
    global time

    clockInterval = parameters['clockInterval'] + 1

    newTime = time_ns()

    deltaTime = (newTime - time) / (clockInterval / 60)

    time = newTime

    # time agnle is 1 hz saled with clock
    timeAngle = parameters['timeAngle']
    timeAngle += deltaTime / 1e+9 * pi * 2
    # convert to degrees then do mod 360 and convert back to radians
    timeAngle = ((timeAngle * degreesRatio) % 360) * radiansRatio

    parameters['timeAngle'] = timeAngle


    for i in range(3):
        # load start value
        stVal = parameters['val'][i]
        maxVal = parameters['maxval'][i]
        # i have no clue how to get constant and trig to work together pls help

        # trig
        trig = parameters['trig'][i]
        if trig != 'none':
            frequencyResponce = parameters['frequency'][i] / 60
            amplitudeResponce = parameters['amplitude'][i]
        if trig == 'sin':
            stVal = sin(timeAngle * frequencyResponce) * amplitudeResponce
        elif trig == 'cos':
            stVal = cos(timeAngle * frequencyResponce) * amplitudeResponce
        elif trig == 'tan':
            stVal = tan(timeAngle * frequencyResponce) * amplitudeResponce

        # constant
        if parameters['constant'][i]:
            fnespeed = parameters['fine speed'][i]
            csespeed = parameters['corse speed'][i]
            amount = (csespeed + (fnespeed / 60)) * (deltaTime / 1e+8)
            if parameters['direction'][i]:
                stVal += amount
            else:
                stVal -= amount

        stVal %= maxVal
        parameters['val'][i] = stVal

def runtest(*args):
    flags['test'] = True

def random_color():
    r = random.randint(50, 255)
    g = random.randint(50, 255)
    b = random.randint(50, 255)
    return (b, g, r)

def ceil_power_of_base(n,base):
    exp = log(n, base)
    exp = ceil(exp)
    return base**exp

def test():
    windowName = 'CPU performance plot'

    delay1 = .2
    delay2 = .0002

    maxSize = parameters['maxSize']
    inc = 1
    iterations = int(maxSize / inc)
    maxSize = iterations * inc

    font_scale = 0.5
    font = cv2.FORMATTER_FMT_DEFAULT

    ((txt_w, txt_h), _) = cv2.getTextSize('0', font, font_scale, 1)

    graphInc = int(maxSize / 10)

    maxSize = graphInc * 10

    margin = int(txt_h * 3)

    yAxisWidth = 250
    labelHeight = 90

    graphHeight = 700
    graphWidth = maxSize

    imageWidth = graphWidth + yAxisWidth + graphInc
    imageHeight = graphHeight + labelHeight * 2

    labelTextHeight = int(labelHeight * 2/3)
    growthPlot = np.zeros((imageHeight,imageWidth,3),dtype=np.uint8)
    spiralFrames = []
    timetable = []
    flags['timing'] = True
    # mabe test other parameters as well?
    # incriment, lineWidth
    tempWidth = parameters['image width']
    for i in range(iterations):
        showProgress((i+1)/iterations,'Benchmark')
        calculateMovement()
        times,names,frame = spiral()
        # get more mesurments
        parameters['image width'] = (i * inc) + 1
        moreTimes = [spiral()[0] for j in range(20)]
        # add in first mesurement
        moreTimes.append(times)
        # average them all
        times = np.average(moreTimes, axis=0)
        # save times
        timetable.append(times)
        spiralFrames.append(frame)
    flags['timing'] = False
    parameters['image width'] = tempWidth

    maxVal = max(max(x) for x in timetable)

    base = 2
    maxPlot = ceil_power_of_base(maxVal,base)
    Ydiv = ceil_power_of_base(maxPlot / 10,base)
    maxPlot = ceil(maxVal / Ydiv) * Ydiv

    numYs = int(maxPlot / Ydiv) + 1

    for i in range(numYs):
        y = graphHeight - int((i * Ydiv)/maxPlot * graphHeight) + labelHeight
        left = (yAxisWidth,y)
        right = (imageWidth - 1 - graphInc,y)
        if i == 0:
            color = [0xFF,0xFF,0xFF]
        else:
            color = [0x60,0x60,0x60]
        cv2.line(growthPlot,left,right,color)
        text = str(i * Ydiv) + 'ns'
        ((txt_w, txt_h), _) = cv2.getTextSize(text, font, font_scale, 1)
        cv2.putText(growthPlot,text,(yAxisWidth - txt_w,y - 2),font,font_scale,color,1)
        cv2.imshow(windowName,growthPlot)
        cv2.waitKey(1)
        sleep(delay1)

    numXs = int(maxSize / graphInc) + 1

    for i in range(numXs):
        x = yAxisWidth + (i * graphInc)
        top = (x,labelHeight)
        bottom = (x,graphHeight + labelHeight)
        if i == 0:
            color = [0xFF,0xFF,0xFF]
        else:
            color = [0x60,0x60,0x60]
        cv2.line(growthPlot,top,bottom,color)
        text = str(i * graphInc) + 'px'
        ((txt_w, txt_h), _) = cv2.getTextSize(text, font, font_scale, 1)
        scale = font_scale
        while txt_w > graphInc:
            scale -= 0.05
            ((txt_w, txt_h), _) = cv2.getTextSize(text, font, scale, 1)
        textHeight = graphHeight + labelHeight + txt_h + 2
        cv2.putText(growthPlot,text,(x,textHeight),font,scale,color,1)
        cv2.imshow(windowName,growthPlot)
        cv2.waitKey(1)
        sleep(delay1)

    colors = []

    pos = margin + yAxisWidth

    for i in range(len(names)):
        name = names[i]
        color = random_color()
        colors.append(color)
        cv2.putText(growthPlot,name,(pos,labelTextHeight),font,font_scale,color,1)
        ((txt_w, txt_h), _) = cv2.getTextSize(name, font, font_scale, 1)
        pos += txt_w + 20
        print(name)

    prevPoints = [None]*len(names)

    for j in range(iterations):
        for i in range(len(names)):
            color = colors[i]
            value = timetable[j][i]
            y = graphHeight - int(value/maxPlot * graphHeight) + labelHeight
            x = (j * inc) + yAxisWidth + 1
            point = (x,y)
            if j > 0:
                cv2.line(growthPlot,prevPoints[i],point,color)
            prevPoints[i] = point
        export = growthPlot.copy()
        if export.shape[0] != spiralFrames[j].shape[0]:
            difference = export.shape[0] / spiralFrames[j].shape[0]
            width = int(difference * spiralFrames[j].shape[1])
            height = export.shape[0]
            dim = (width,height)
            spiralFrames[j] = cv2.resize(spiralFrames[j],dim,interpolation=cv2.INTER_AREA)
        export = np.concatenate((export, spiralFrames[j]),axis=1)
        cv2.imshow(windowName,export)
        cv2.waitKey(1)
        sleep(delay2)

    flags['test'] = False
    pass

def save(*args):
    # lol why did i even make this a funciton, or right i had to DX
    flags['save'] = True

def createGif():
    maxFrames = 500
    first = None
    image_count = 0
    frames = []
    while True:
        frame = spiral()
        equal = np.array_equal(first,frame)
        if equal or image_count > maxFrames:
            print("Saving GIF file")
            with imageio.get_writer("spiral.gif", mode="I") as writer:
                for idx, frame in enumerate(frames):
                    print("Adding frame to GIF file: ", idx + 1)
                    writer.append_data(frame)
            flags['save'] = False
            return
        else:
            frames.append(frame)
            image_count += 1
        if np.array_equal(first,None):
            first = frame

def offsetPoint(point,offset):
    for i in range(2):
        point[i] += offset[i]


def showProgress(value,title=None):

    global windowName

    height = 400
    width = 300

    show = np.zeros((height,width,3),dtype=np.uint8)

    center = [int(width/2),int(height/2)]

    barWidth = 10

    radii = [int(width * 1/3),int(width * 1/3) - barWidth]

    uiColor = parameters['uiColor']

    font_scale = 10

    font = cv2.FORMATTER_FMT_DEFAULT

    if title != None:
        ((txt_w, txt_h), _) = cv2.getTextSize(title, font, font_scale, 3)
        scale = font_scale
        while txt_w > width - 10:
            scale -= 0.01
            ((txt_w, txt_h), _) = cv2.getTextSize(title, font, scale, 3)
        cv2.putText(show,title,(center[0] - int(txt_w / 2),txt_h + 10),font,scale,uiColor,3)

    text = str(int(value * 100)) + '%'
    ((txt_w, txt_h), _) = cv2.getTextSize(text, font, font_scale, 3)
    scale = font_scale
    while txt_w > (radii[1] * 2) - 5:
        scale -= 0.01
        ((txt_w, txt_h), _) = cv2.getTextSize(text, font, scale, 3)
    cv2.putText(show,text,(center[0] - int(txt_w / 2),center[1] + int(txt_h / 2)),font,scale,uiColor,3)

    valueAngle = value * 2 * pi

    step = 5

    points = [None,None]

    for i in range(0,360 + 1,step):
        angle = i * radiansRatio
        trig = [cos(angle),sin(angle)]

        # get points for these two radii
        points[0] = [[int(center[k] + trig[k] * radii[j]) for k in range (2)] for j in range(2)]
        if i > 0:
            # join the last to points in reverse order with the currect two points into an array
            poly = array(points[0] + [points[1][1],points[1][0]])
            # fill in portion that is value or below, else only fill outline
            if angle <= valueAngle:
                show = cv2.fillPoly(show,[poly],uiColor)
            else:
                cv2.line(show,poly[0],poly[3],uiColor)
                cv2.line(show,poly[1],poly[2],uiColor)
        # save the points
        points[1] = points[0]

    cv2.imshow(windowName,show)
    cv2.waitKey(1)

if __name__ == '__main__':
    createControlls()

    while True:
        calculateMovement()
        frame = spiral()
        cv2.imshow(windowName,frame)
        cv2.waitKey(1)

        # subroutines
        if flags['save']:
            createGif()
        if flags['test']:
            test()

# todo:
# constant performance graph parameter
# fix ageraging in test : DONE
# make progress bar function
# optomise time graph to use new poly method. mabe generalise it into a function to use more than once? - NO
