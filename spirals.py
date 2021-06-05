import cv2
import imageio
import numpy as np
from numpy import array
from math import sin, cos, tan, pi, sqrt, ceil, log, degrees, radians
from time import time_ns, sleep
from random import randint
# from str import join

flags = {
    'test':False,
    'save':False,
    'animateGraph':False
}

# defaults
parameters = {
    'rotateX':20,
    'rotateY':22,
    'angle manipulation':'exp',
    'grid':10,
    'graphLength':300,
    'graphSize':600,
    'graphcolors':[None],
    'maxSize':100,
    'uiColor':[0xDD,0xDD,0xDD],
    'graph':{'spiral fill':True,'phase':True,'spiral trace':False,'unit':True,'time graph':True,'colorspace':True,'performance':True},
    'colorspace':'BGR',
    'clockInterval':100,
    'timeAngle':0,
    'image width':500,
    'lineWidth':0,
    'exponent':100,
    'lineWidth Interpolation':'distance',
    'radius interpolation':'linear',
    'incriment':0,
    'fillType':'polygon',
    'majr axis':0,
    'semi axis':0,
    'angle':0,
    'phase threshold':25,
    'colorspace threshold':25,
    'phase angle':[0,0,0],
    'corse angle dif':[1,1,1],
    'fine angle dif':[179,179,179],
    'trig':[None,None,None],
    'direction':[False,False,False],
    'fine speed':[0,0,0],
    'corse speed':[20,10,4],
    'constant':[True,True,True],
    'val':[0,0,0],
    'maxval':[0xFF,0xFF,0xFF],
    'frequency':[1,0,0],
    'amplitude':[1,0,0]
}

spiraltime = 0
processtime = 0
fpstime = 0

# values that will never change, ever, mostly just math constants
windowName = 'spiral'
fullRadians = 2 * pi

signature = None

# lists that get compiled once at the beginning and when signature (4 key parameters) changes
# otherwise they are just read from
# aka, this is data that does not change from frame to frame so make it once and just read from the list

fillpolyList = []
fillposList = []
fillangleList = []
fillthicknessList = []

phaseposList = []
phaseangleList = []

def start():
    global processtime
    processtime = time_ns()

def stop():
    global processtime
    delta = time_ns() - processtime
    processtime = time_ns()
    return delta

def nonZero(arg: int):
    if arg == 0:
        arg = 1
    return arg

def radius(RintLinear,i,widthratio,exp):
    if RintLinear:
        r = i * widthratio
    else:
        r = (i ** exp)
    return r

def spiral():

    beginTime = time_ns()
    start()

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

    # width of the picture
    imagewidth = nonZero(parameters['image width'])

    # amount to incriment each fill cycle
    incriment = nonZero(parameters['incriment'])

    # calculations to keep the plot within the image
    lineWidth = nonZero(parameters['lineWidth'])

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

    # color diferential per angle change
    fneDifs = [parameters['fine angle dif'][i] for i in range(3)]
    cseDifs = [parameters['corse angle dif'][i] for i in range(3)]
    chandiffs = [(cseDifs[j] * 180  + fneDifs[j] - 180) / maxchannels[j] for j in range(3)]

    # parameters for ellipsoids
    MA = nonZero(parameters['majr axis'])
    SA = nonZero(parameters['semi axis'])
    rotation = parameters['angle']

    # distance from edge for each graph (other than fill)
    graphOffset = int(image.shape[1] / 20)

    # conversion factors
    widthratio = lineWidth / 360
    incrimentRatio = 360/incriment

    # define colors
    uiColor = parameters['uiColor']

    # list of what graphs to graph
    graphlist = parameters['graph']

    colorspace = parameters['colorspace']

    LintConstant = parameters['lineWidth Interpolation'] == 'constant'

    RintLinear = parameters['radius interpolation'] == 'linear'

    exp = parameters['exponent'] / 360

    rotX = parameters['rotateX'] / 360

    rotY = parameters['rotateY'] / 360

    aManipLinear = parameters['angle manipulation'] == 'linear'

    times = []
    names = []

    global fillpolyList
    global fillposList
    global fillangleList
    global fillthicknessList
    global phaseposList
    global phaseangleList
    fillType = parameters['fillType']

    global signature

    keyParams = (imagewidth,incriment,lineWidth,fillType,LintConstant,RintLinear,exp,rotX,rotY,aManipLinear)

    mismatch = signature != keyParams

    if mismatch:
        fillpolyList = []
        fillposList = []
        fillangleList = []
        fillthicknessList = []
        phaseposList = []
        phaseangleList = []
        signature = keyParams

    times.append(stop())
    names.append('parameters')

    if graphlist['spiral fill']:
        # storage for previous positions
        prev = None
        if mismatch:
            if fillType == 'polygon':
                points = [None]*2
            for i in range(0,(360 * numTurns)+1,incriment):
                # current fillRadius for fill
                r = radius(RintLinear,i,widthratio,exp)
                # limit radius to edge of fill graph
                if RintLinear:
                    if r > maxR:
                        r = maxR
                # angle of fill in radians
                angle = radians(i)
                # trig calculations
                if aManipLinear:
                    trig = [cos(angle + rotX),sin(angle + rotY)]
                else:
                    trig = [cos(angle * rotX),sin(angle * rotY)]
                # calculate color
                color = [(stVals[j] + i * chandiffs[j]) % maxchannels[j] for j in range(3)]
                # filling the graph
                if fillType == 'polygon':
                    # Radius, and one inwards
                    radii = [r,r - lineWidth]
                    if radii[0] == radii[1]:
                        # polygon of zero size... skip it
                        continue
                    # no negitive radius shenanagans
                    if radii[1] < 0:
                        radii[1] = 0
                    # get points for these two radii
                    points[0] = [[int(origin + trig[l] * radii[k]) for l in range (2)] for k in range(2)]
                    if i > 0:
                        if(points[0] == points[1]):
                            # polygon of zero size... skip it
                            continue
                        # join the last to points in reverse order with the currect two points into an array
                        poly = array(points[0] + [points[1][1],points[1][0]])
                        fillpolyList.append(poly)
                        fillangleList.append(i)
                        # apply polygon
                        image = cv2.fillPoly(image,[poly],color)
                    # save the points
                    points[1] = points[0]
                else:
                    # polar to cartisian conversion
                    pos = [int(origin + trig[j] * r) for j in range(2)]
                    if i > 0:
                        if LintConstant:
                            thickness = lineWidth
                        else:
                            thickness = int(sqrt(abs(pos[0] - prev[0])**2 + abs(pos[1] - prev[1])**2))
                            if thickness == 0:
                                thickness = 1
                            fillthicknessList.append(thickness)
                        fillposList.append([prev,pos])
                        fillangleList.append(i)
                        if fillType == 'line':
                            cv2.line(image,prev,pos,color,thickness)
                        elif fillType == 'rectangle center':
                            cv2.rectangle(image,prev,pos,color,thickness)
                        elif fillType == 'circle':
                            thickness = int(thickness/2)
                            cv2.circle(image,pos,thickness,color,-1)
                        elif fillType == 'ellipse':
                            thickness
                            cv2.ellipse(image, (pos, (thickness/MA,thickness/SA),i + rotation), color, -1)
                    prev = pos
        else:
            if fillType == 'polygon':
                for i in range(len(fillangleList)):
                    deg = fillangleList[i]
                    poly = fillpolyList[i]
                    color = [(stVals[j] + deg * chandiffs[j]) % maxchannels[j] for j in range(3)]
                    image = cv2.fillPoly(image,[poly],color)
            else:
                for i in range(len(fillangleList)):
                    deg = fillangleList[i]
                    color = [(stVals[j] + deg * chandiffs[j]) % maxchannels[j] for j in range(3)]
                    [prev,pos] = fillposList[i]
                    if LintConstant:
                        thickness = lineWidth
                    else:
                        thickness = fillthicknessList[i]
                    if fillType == 'line':
                        cv2.line(image,prev,pos,color,thickness)
                    elif fillType == 'rectangle center':
                        cv2.rectangle(image,prev,pos,color,thickness)
                    elif fillType == 'circle':
                        thickness = int(thickness/2)
                        cv2.circle(image,pos,thickness,color,-1)
                    elif fillType == 'ellipse':
                        cv2.ellipse(image, (pos, (thickness/MA,thickness/SA),i + rotation), color, -1)
        times.append(stop())
        names.append('spiral fill')

    # spiral graph
    if graphlist['spiral trace']:
        prev = None
        for i in range(0,360 * numTurns,incriment):
            r = radius(RintLinear,i,widthratio,exp)
            if RintLinear:
                if r > maxR:
                    r = maxR
            if r < 0:
                r = 0
            angle = radians(i)
            if aManipLinear:
                trig = [cos(angle + rotX),sin(angle + rotY)]
            else:
                trig = [cos(angle * rotX),sin(angle * rotY)]
            point  = [int(origin + trig[j] * r) for j in range (2)]
            if i > 0:
                cv2.line(image,prev,point,uiColor)
            prev = point
        times.append(stop())
        names.append('spiral trace')

    # time cycle graph
    if graphlist['time graph']:

        # size
        timeRadius = int(image.shape[1] / 9)
        # position
        center = int((image.shape[1]) - timeRadius - graphOffset),int(timeRadius + graphOffset)

        # width of each loop
        width = timeRadius / 6
        # fillRadius of each loop end
        radii = [(timeRadius - i * width) for i in range(5)]

        global fpstime
        fps = int(1e+9 / (time_ns() - fpstime))
        fpstime = time_ns()

        fps = str(fps) + 'FPS'

        font_scale = 10

        font = cv2.FORMATTER_FMT_DEFAULT

        txt_w = cv2.getTextSize(fps, font, font_scale, 1)[0][0]
        scale = font_scale
        while txt_w > (radii[4] * 2) - 4:
            scale -= 0.01
            ((txt_w, txt_h), _) = cv2.getTextSize(fps, font, scale, 1)
        cv2.putText(image,fps,(center[0] - int(txt_w / 2),center[1] + int(txt_h / 2)),font,scale,uiColor,1)

        chanangles = [(stVals[i] / maxchannels[i]) * fullRadians for i in range(3)]
        chanangles.append(parameters['timeAngle'])
        # circular gradiants
        step = 10
        points = [None,None]
        plotted = [False] * 4
        for j in range(0,361,step):
            angleRatio = j / 360
            angle = radians(j)
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

        times.append(stop())
        names.append('time graph')

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
        if parameters['colorspace'] == 'HSV':
            div = [1,1,3]
        else:
            div = [3,3,3]
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
                    color = [color[i] / div[i] for i in range(3)]
                    for l in range(inc):
                        image[y][x + l] = color
        # calculate the radius of the circle of points over the threshhold given the number of points (area)
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
        cv2.rectangle(image,tl,br,uiColor,1)
        short = width < 360
        if short:
            ratio = 360 / width
        end = width * numTurns
        times.append(stop())
        names.append('phase pre')
        for i in range(0,end,inc):
            if mismatch:
                if short:
                    deg = int(i * ratio)
                else:
                    deg = i
                x = left + int(i % width)
                y = top + int(i / width)
                pos = (x,y)
                phaseposList.append(pos)
                phaseangleList.append(deg)
            else:
                index = int(i / inc)
                (x,y) = phaseposList[index]
                deg = phaseangleList[index]
            channels = [(stVals[j] + deg * chandiffs[j]) % maxchannels[j] for j in range(3)]
            plotted = False
            for j in range(3):
                if channels[j] > maxchannels[j] - threshold:
                    plotted = True
                    image[y][x][j] = maxchannels[j] - 1
            if not plotted:
                if deg % 180 == 0:
                    image[y][x] = [0x80,0x80,0x80]
                elif deg % 90 == 0:
                    image[y][x] = [0x80,0,0x80]
                elif deg % 45 == 0:
                    image[y][x] = [0x80,0x80,0]
                elif deg % 30 == 0:
                    image[y][x] = [0,0x80,0x80]
                else:
                    image[y][x] = [channels[0] / div,channels[1] / div,channels[2] / div]
        times.append(stop())
        names.append('phase loop')

    # unit circle
    if graphlist['unit']:
        # importiant angles
        pioversix = pi/6
        pioverfour = pi/4
        angleList = [i * pioversix for i in range(6)] + [i * pioverfour for i in range(1,4,2)]
        for i in range(len(angleList)):
            angle = angleList[i]
            if aManipLinear:
                trig = [cos(angle + rotX),sin(angle + rotY)]
            else:
                trig = [cos(angle * rotX),sin(angle * rotY)]
            end1 = [int(origin + trig[j] * fillRadius) for j in range(2)]
            end2 = [int(origin - trig[j] * fillRadius) for j in range(2)]
            if i < 6:
                cv2.line(image,end1,end2,[0xFF,0,0xFF])
            else:
                cv2.line(image,end1,end2,[0xFF,0xFF,0])

        times.append(stop())
        names.append('unit')

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

    times.append(time_ns() - beginTime)
    names.append('total time')

    return image,times,names

def change(*args):
    if len(args[1]) > 2:
        suffix = ['blue','green','red']
        print('change parameters[{:10}][{:10}] = {:10}'.format(args[1][0],suffix[args[1][1]],args[1][2]))
        parameters[args[1][0]][args[1][1]] = args[1][2]
    else:
        print('change parameters[{:10}] = {:10}'.format(args[1][0],args[1][1]))
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
    print('check  parameters[{:10}][{:10}] = {:10}'.format(args[1][0],args[1][1],args[0]))
    parameters[args[1][0]][args[1][1]] = args[0]
    pass

# trackbar callback, one callback to rule them all
def tbcb(key,value):
    print(key,value)
    if type(key) == list:
        parameters[key[0]][key[1]] = value
    else:
        parameters[key] = value

def runtest(*args):
    flags['test'] = True

def save(*args):
    # lol why did i even make this a funciton, or right i had to DX
    flags['save'] = True

def reset(*args):
    # todo: make a reset function that works
    pass

# create track bar
def makeTbr(tname,wname,dval,mval,callback):
    # print('{:25s} {:10s} {:<9d} {:<9d} {}'.format(tname,wname,dval,mval,callback))
    showProgress(title = 'loading trackbar ' + tname)
    cv2.createTrackbar(tname,wname,dval,mval,callback)
# create track bar
def makeBtn(name,callback,data,type,default):
    # print('{:25s} {:10s} {:<9d} {:<9d} {}'.format(tname,wname,dval,mval,callback))
    showProgress(title = 'loading button ' + name)
    cv2.createButton(name,callback,data,type,default)

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
        if i % 2 == 0:
            showProgress((i+1)/height,'building controls image')
        for j in range(blank.shape[1]):
            for k in range(3):
                blank[i][j][k] += j / 4
                blank[i][j][k] *= 3/7

    blank = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)

    font = cv2.FORMATTER_FMT_DEFAULT

    cv2.putText(blank,windowName,(30,y1),font,size1,(100, 0, 0),3)
    cv2.putText(blank,windowName,(30,y1),font,size1,(200, 0, 0),2)

    cv2.putText(blank,'ctrl + p for more controls',(30,y2),font,size2,(100, 0, 0),3)
    cv2.putText(blank,'ctrl + p for more controls',(30,y2),font,size2,(200, 0, 0),2)

    cv2.imshow(windowName,blank)
    cv2.waitKey(1)

    suffix = ['blue','green','red']

    general_configs = {
        'image width':1000,
        'clockInterval':1000,
        'incriment':360,
        'lineWidth':256,
        'majr axis':30,
        'semi axis':30,
        'angle':360,
        'phase threshold':360,
        'colorspace threshold':255,
        'maxSize':2000,
        'grid':50,
        'graphLength':2000,
        'graphSize':2000,
        'exponent':360,
        'rotateX':3600,
        'rotateY':3600
   }

    for cfg in general_configs:
        makeTbr(cfg,windowName,parameters[cfg],general_configs[cfg],lambda v, cfg=cfg :tbcb(cfg,v))

    channel_cfgs = {'frequency':60,'amplitude':256,'phase angle':360}

    widecfgs = {'speed':[60,60],'angle dif':[30,360]}

    buttonTypes = ['direction','constant']

    trigs = ['sin','cos','tan',None]

    for color in suffix:
        a = suffix.index(color)
        for wcfg in widecfgs:
            makeTbr(color + ' corse ' + wcfg,windowName,parameters['corse ' + wcfg][a],widecfgs[wcfg][0],lambda v, a=a, wcfg=wcfg:tbcb(['corse ' + wcfg,a],v))
            makeTbr(color + ' fine '  + wcfg,windowName,parameters['fine '  + wcfg][a],widecfgs[wcfg][1],lambda v, a=a, wcfg=wcfg:tbcb(['fine '  + wcfg,a],v))

        for cfg in channel_cfgs:
            makeTbr(color + ' ' + cfg,windowName,parameters[cfg][a],channel_cfgs[cfg],lambda v, a=a, cfg=cfg:tbcb([cfg,a],v))

        for btype in buttonTypes:
            makeBtn(color + ' ' + btype,check,[btype,a],1,parameters[btype][a])

        for trig in trigs:
            if trig == None:
                name = color + 'None'
            else:
                name = color + ' ' + trig
            makeBtn(name,change,['trig',a,trig] ,2,parameters['trig'][a] == trig)

    filltypes = ['rectangle center','line','circle','ellipse','polygon']

    key = 'fillType'
    for fill in filltypes:
        makeBtn(fill + ' ' + key,change,[key,fill],2,parameters[key] == fill)

    graphTypes =  ['spiral fill','phase','spiral trace','unit','time graph','colorspace','performance']

    key = 'graph'
    for graph in graphTypes:
        makeBtn(graph + ' ' + key,check,[key,graph],1,parameters[key][graph])

    colorspaces = ['BGR','HSV','LUV','YUV','LAB']

    key = 'colorspace'
    for code in colorspaces:
        makeBtn(code + ' ' + key,change,[key,code],2,parameters[key] == code)

    Lints = ['constant','distance']

    key = 'lineWidth Interpolation'
    for Lint in Lints:
        makeBtn(Lint + ' ' + key,change,[key,Lint],2,parameters[key] == Lint)

    Rints = ['linear','exp']

    key = 'radius interpolation'
    for Rint in Rints:
        makeBtn(Rint + ' ' + key,change,[key,Rint],2,parameters[key] == Rint)

    aManips = ['linear','exp']

    key = 'angle manipulation'
    for manip in aManips:
        makeBtn(manip + ' ' + key,change,[key,manip],2,parameters[key] == manip)

    cv2.createButton('reset',reset)

    cv2.createButton('save',save)

    cv2.createButton('run benchmark test',runtest)

def calculateMovement():
    global spiraltime

    clockInterval = parameters['clockInterval'] + 1

    newTime = time_ns()

    deltaTime = (newTime - spiraltime) / (clockInterval / 60)

    spiraltime = newTime

    # time agnle is 1 hz saled with clock
    timeAngle = parameters['timeAngle']
    timeAngle += deltaTime / 1e+9 * pi * 2
    # convert to degrees then do mod 360 and convert back to radians
    timeAngle = radians(degrees(timeAngle) % 360)

    parameters['timeAngle'] = timeAngle

    for i in range(3):
        # load start value
        stVal = parameters['val'][i]
        maxVal = parameters['maxval'][i]
        # i have no clue how to get constant and trig to work together pls help

        # trig
        trig = parameters['trig'][i]
        if trig != None:
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

def random_color():
    r = randint(50, 255)
    g = randint(50, 255)
    b = randint(50, 255)
    return (b, g, r)

def ceil_power_of_base(n,base):
    exp = log(n, base)
    exp = ceil(exp)
    return base**exp

def fitWidth(text,space):
    font_scale = 10
    font = cv2.FORMATTER_FMT_DEFAULT
    scale = font_scale
    txt_w = cv2.getTextSize(text, font, font_scale, 1)[0][0]
    while txt_w > space:
        scale -= 0.05
        txt_w = cv2.getTextSize(text, font, scale, 1)[0][0]
    return scale

def fitHeight(text,space):
    font_scale = 10
    font = cv2.FORMATTER_FMT_DEFAULT
    scale = font_scale
    txt_h = cv2.getTextSize(text, font, font_scale, 1)[0][1]
    while txt_h > space:
        scale -= 0.05
        txt_h = cv2.getTextSize(text, font, scale, 1)[0][1]
    return scale

def graph(table,inc,names,units):

    uiColor = parameters['uiColor']

    animate = flags['animateGraph']

    numXs = parameters['grid'] + 1
    base = 2
    graphHeight = parameters['graphSize']

    windowName = 'CPU performance plot'

    delay1 = .2
    delay2 = .0002

    tableLength = len(table[0])
    tableHeight = len(table)

    if tableHeight > graphHeight:
        graphWidth = tableHeight
    else:
        graphWidth = graphHeight

    graphInc = int(graphWidth / numXs)
    graphWidth = graphInc * numXs

    maxVal = max(max(x) for x in table)

    font = cv2.FORMATTER_FMT_DEFAULT

    yAxisWidth = cv2.getTextSize('{:e}'.format(maxVal), font, 1, 1)[0][0] + 10

    extra = cv2.getTextSize(names[names.index(max(names))], font, 1, 1)[0][0] + 10

    if graphInc > extra:
        extra = 0
    else:
        extra = extra - graphInc

    labelHeight = 100

    imageWidth = graphWidth + yAxisWidth + graphInc + extra
    imageHeight = graphHeight + labelHeight * 2

    graphImage = np.full((imageHeight,imageWidth,3),[10,10,10],dtype=np.uint8)

    cv2.rectangle(graphImage,(yAxisWidth,labelHeight),(yAxisWidth + graphWidth,labelHeight + graphHeight),[0,0,0],-1)

    tableMax = ceil_power_of_base(maxVal,base)
    Ydiv = ceil_power_of_base(tableMax / numXs,base)
    tableMax = ceil(maxVal / Ydiv) * Ydiv

    numYs = int(tableMax / Ydiv)

    gridHeight = int(graphHeight / numYs)

    for i in range(numYs + 1):
        y = graphHeight - int((i * Ydiv)/tableMax * graphHeight) + labelHeight
        left = (yAxisWidth,y)
        right = (yAxisWidth + graphWidth,y)
        if i == 0:
            color = uiColor
        else:
            color = [uiColor[j] / 3 for j in range(3)]
        cv2.line(graphImage,left,right,color)
        text = '{:e}'.format((i * Ydiv)) + 'ns'
        scale = min(fitWidth(text,yAxisWidth - 7),fitHeight(text,gridHeight - 5))
        txt_w = cv2.getTextSize(text, font,scale, 1)[0][0]
        cv2.putText(graphImage,text,(yAxisWidth - txt_w,y - 2),font,scale,color,1)
        if(animate):
            cv2.imshow(windowName,graphImage)
            cv2.waitKey(1)
            sleep(delay1)

    for i in range(numXs + 1):
        x = yAxisWidth + (i * graphInc)
        top = (x,labelHeight)
        bottom = (x,graphHeight + labelHeight)
        if i == 0:
            color = [0xFF,0xFF,0xFF]
        else:
            color = [0x60,0x60,0x60]
        cv2.line(graphImage,top,bottom,color)
        text = str(i * graphInc) + units
        scale = fitWidth(text,graphInc)
        txt_h = cv2.getTextSize(text, font, scale, 1)[0][1]
        textHeight = graphHeight + labelHeight + txt_h + 2
        cv2.putText(graphImage,text,(x,textHeight),font,scale,color,1)
        if(animate):
            cv2.imshow(windowName,graphImage)
            cv2.waitKey(1)
            sleep(delay1)

    if len(parameters['graphcolors']) != tableLength:
        colors = [random_color() for i in range(tableLength)]
        parameters['graphcolors'] = colors
    else:
        colors = parameters['graphcolors']

    labelTextHeight = int(labelHeight * 1/3)

    averageTextHeight = int(labelHeight * 3/4)

    pos = yAxisWidth

    LabelWidth = imageWidth - yAxisWidth

    scale = fitWidth('  '.join(names),LabelWidth - 10)

    averages = np.average(table,axis=0)

    for i in range(tableLength):
        cv2.putText(graphImage,names[i],(pos,labelTextHeight),font,scale,colors[i],1)
        (txt_w, txt_h) = cv2.getTextSize(names[i] + '  ', font, scale, 1)[0]
        averagePos = (pos,averageTextHeight)
        pos += txt_w
        if(animate):
            cv2.imshow(windowName,graphImage)
            cv2.waitKey(1)
            sleep(delay1)
        prevPoint = None
        for j in range(tableHeight):
            color = colors[i]
            value = table[j][i]
            y = graphHeight - nonZero(int(value/tableMax * graphHeight)) + labelHeight
            if tableHeight != graphWidth:
                x = nonZero(int(j / tableHeight * graphWidth) + yAxisWidth)
            else:
                x = nonZero(j + yAxisWidth)
            point = (x,y)
            if j > 0:
                cv2.line(graphImage,prevPoint,point,color)
            prevPoint = point
            if(animate):
                cv2.imshow(windowName,graphImage)
                cv2.waitKey(1)
                sleep(delay2)

        x = yAxisWidth + graphWidth
        cv2.line(graphImage,(x,y),point,uiColor)
        cv2.putText(graphImage,names[i],(x,y),font,fitWidth(names[i],imageWidth - x - 10),colors[i],1)
        if(animate):
            cv2.imshow(windowName,graphImage)
            cv2.waitKey(1)
            sleep(delay1)
        text = '{:.2e}'.format(averages[i])
        cv2.putText(graphImage,text,averagePos,font,fitWidth(text,txt_w - 4),colors[i],1)
        if(animate):
            cv2.imshow(windowName,graphImage)
            cv2.waitKey(1)
            sleep(delay1)

    if(not animate):
        cv2.imshow(windowName,graphImage)
        cv2.waitKey(1)
        sleep(delay2)

def test():

    maxSize = parameters['maxSize']
    inc = 1
    iterations = int(maxSize / inc)
    names = spiral()[2]
    table = []

    # mabe test other parameters as well?
    # incriment, lineWidth
    tempWidth = parameters['image width']
    for i in range(iterations):
        showProgress((i+1)/iterations,'Benchmark')
        calculateMovement()
        parameters['image width'] = (i * inc) + 1
        # get multiple mesurments
        # average them all and put them into table
        table.append(np.average([spiral()[1] for j in range(20)], axis=0))

    parameters['image width'] = tempWidth
    flags['animateGraph'] = True
    graph(table,inc,names,'px')
    flags['animateGraph'] = False
    flags['test'] = False
    pass

def createGif():
    global windowName
    maxFrames = 150
    first = None
    image_count = 0
    frames = []
    while True:
        calculateMovement()
        frame = spiral()[0]
        cv2.imshow(windowName,frame)
        cv2.waitKey(1)
        showProgress((image_count + 1) / maxFrames, 'saving image')
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


def showProgress(value=None,title=None):

    global windowName

    height = 400
    width = 300
    show = np.zeros((height,width,3),dtype=np.uint8)
    center = [int(width/2),int(height/2)]
    uiColor = parameters['uiColor']
    font_scale = 10
    font = cv2.FORMATTER_FMT_DEFAULT

    if title != None:
        thickness = 2
        ((txt_w, txt_h), _) = cv2.getTextSize(title, font, font_scale, thickness)
        scale = font_scale
        while txt_w > width - 10:
            scale -= 0.01
            ((txt_w, txt_h), _) = cv2.getTextSize(title, font, scale, thickness)
        cv2.putText(show,title,(center[0] - int(txt_w / 2),txt_h + 10),font,scale,uiColor,thickness)

    if value != None:
        barWidth = 10
        radii = [int(width * 1/3),int(width * 1/3) - barWidth]
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
            angle = radians(i)
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

    table = []

    while True:
        calculateMovement()
        graphLength = parameters['graphLength'] + 1
        if parameters['graph']['performance']:
            frame,times,names = spiral()
            table.append(times)
            if len(table[-1]) != len(table[0]):
                table = []
                table.append(times)
            while len(table) > graphLength:
                table.pop(0)
            graph(table,1,names,'frame')
        else:
            if table != []:
                table = []
            frame = spiral()[0]
        cv2.imshow(windowName,frame)
        cv2.waitKey(1)

        # subroutines
        if flags['save']:
            createGif()
        if flags['test']:
            test()

# todo:
# fix calculation for fill radius such that when doing point type fill it adjusts to fit larger fill widths durring distance baced thickness interpolation... that was anoying just to type
# constant performance graph parameter : DONE
# fix ageraging in test : DONE
# make progress bar function : DONE
# optomise time graph to use new poly method. mabe generalise it into a function to use more than once? - NO actually yes
# calculate num turn with exponential radii
