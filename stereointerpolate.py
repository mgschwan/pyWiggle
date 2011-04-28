import cv
import sys
from optparse import OptionParser





# Creates a x/y velocity field on the interval [0,1]
def createVelocityField(velx , vely, delta):
    new_velx = cv.CreateMat(velx.rows, velx.cols, cv.CV_32FC1)
    new_vely = cv.CreateMat(vely.rows, vely.cols, cv.CV_32FC1)
    for x in range (0, velx.cols):
        for y in range (0, velx.rows):    
            new_velx[y,x] = velx[y,x] * delta
            new_vely[y,x] = vely[y,x] * delta

    return (new_velx, new_vely)

#Apply a simple median Filter to each component of the velocity field
#This function is really slow !!
def filterVelocityField(velx,vely,kernel_size=3):
    new_velx = cv.CreateMat(velx.rows, velx.cols, cv.CV_32FC1)
    new_vely = cv.CreateMat(vely.rows, vely.cols, cv.CV_32FC1)
    
    for x in range (kernel_size//2, velx.cols-kernel_size//2):
        for y in range (kernel_size//2, velx.rows-kernel_size//2):    
            values = []
            for dx in range(kernel_size):
                for dy in range(kernel_size):
                     values.append(velx[y-kernel_size//2+dy,x-kernel_size//2+dx])
            values.sort()
            new_velx[y,x] = values[len(values)//2]            

    for x in range (kernel_size//2, vely.cols-kernel_size//2):
        for y in range (kernel_size//2, vely.rows-kernel_size//2):    
            values = []
            for dx in range(kernel_size):
                for dy in range(kernel_size):
                     values.append(vely[y-kernel_size//2+dy,x-kernel_size//2+dx])
            values.sort()
            new_vely[y,x] = values[len(values)//2]            

    return (new_velx,new_vely)


def parseCommandLine(argv):
    parser = OptionParser()
    parser.add_option("-l","--left", action="store", type="string",
            dest="left", help="left image", default = None)
    parser.add_option("-r","--right", action="store", type="string", 
            dest="right", help="right image", default = None)
    parser.add_option("-a","--autoextract", action="store_true", 
            dest="xextract", help="automatically split the left image into two images",
            default=False)
    parser.add_option("-A","--autoextracty", action="store_true", 
            dest="yextract", help="automatically split the left image into two images (vertically)",
            default=False)
    parser.add_option("-m","--maxrange", action="store", type="string", 
            dest="maxrange", help="maximum search range (x,y)",
            default = "32,32")
    parser.add_option("-b","--blocksize", action="store", type="string", 
            dest="blocksize", help="search block size (w,h)",
            default = "32,32")
    parser.add_option("-s","--shiftsize", action="store", type="string", 
            dest="shiftsize", help="distance between search windows (x,y)",
            default = "10,10")
    parser.add_option("-f","--filtersize", action="store", type="int", 
            dest="filtersize", help="size of the median filter",
            default = "0")
    parser.add_option("-c","--crop", action="store", type="string", 
            dest="crop", help="crop the image (x,y,w,h)",
            default = "-1,-1,-1,-1")
    
    parser.add_option("-S","--steps", action="store", type="int", 
            dest="steps", help="number of interpolation steps",
            default = 20)
    


    (options, args) = parser.parse_args(argv)

    arguments = {}
    arguments["steps"] = options.steps
    arguments["left"] = options.left
    arguments["right"] = options.right
    arguments["extract"] = 0
    if options.xextract:
        arguments["extract"] = 1
    elif options.yextract:
        arguments["extract"] = 2

    cfields=options.crop.split(",")
    arguments["cropx"] = int(cfields[0])
    arguments["cropy"] = int(cfields[1])
    arguments["cropw"] = int(cfields[2])
    arguments["croph"] = int(cfields[3])


    mfields = options.maxrange.split(",")
    arguments["maxrange"] = (int(mfields[0]), int(mfields[1]))

    bfields = options.blocksize.split(",")
    arguments["blocksize"] = (int(bfields[0]), int(bfields[1]))

    sfields = options.shiftsize.split(",")
    arguments["shiftsize"] = (int(sfields[0]), int(sfields[1]))

    arguments["filtersize"] = options.filtersize

    return arguments
    

if __name__=="__main__":
    cmdline = parseCommandLine(sys.argv)
        
    bsize = cmdline["blocksize"]
    shiftsize = cmdline["shiftsize"]
    maxrange = cmdline["maxrange"]
    steps = cmdline["steps"]
    filtersize = cmdline["filtersize"]


    left = None
    right = None
    left_orig = None     #The original images
    right_orig = None    #they may be color images
    cropsize = [cmdline["cropx"],cmdline["cropy"],cmdline["cropw"],cmdline["croph"]]


    if cmdline["extract"] == 1:
        tmp = cv.LoadImageM(cmdline["left"], cv.CV_LOAD_IMAGE_GRAYSCALE)
        left = cv.GetSubRect(tmp, (0,0,int(tmp.cols/2), tmp.rows))
        right = cv.GetSubRect(tmp, (int(tmp.cols/2),0,int(tmp.cols/2), tmp.rows))
        tmp_orig = cv.LoadImageM(cmdline["left"], cv.CV_LOAD_IMAGE_UNCHANGED)
        left_orig = cv.GetSubRect(tmp_orig, (0,0,int(tmp.cols/2), tmp.rows))
        right_orig = cv.GetSubRect(tmp_orig, (int(tmp.cols/2),0,int(tmp.cols/2), tmp.rows))

    elif cmdline["extract"] == 2:
        tmp = cv.LoadImageM(cmdline["left"], cv.CV_LOAD_IMAGE_GRAYSCALE)
        left = cv.GetSubRect(tmp, (0,0,tmp.cols, int(tmp.rows/2)))
        right = cv.GetSubRect(tmp, (0,int(tmp.rows/2),tmp.cols, int(tmp.rows/2)))
        tmp_orig = cv.LoadImageM(cmdline["left"], cv.CV_LOAD_IMAGE_UNCHANGED)
        left_orig = cv.GetSubRect(tmp, (0,0,tmp.cols, int(tmp.rows/2)))
        right_orig = cv.GetSubRect(tmp, (0,int(tmp.rows/2),tmp.cols, int(tmp.rows/2)))

    else:
        left = cv.LoadImageM(cmdline["left"], cv.CV_LOAD_IMAGE_GRAYSCALE)
        right = cv.LoadImageM(cmdline["right"], cv.CV_LOAD_IMAGE_GRAYSCALE)
        left_orig = cv.LoadImageM(cmdline["left"], cv.CV_LOAD_IMAGE_UNCHANGED)
        right_orig = cv.LoadImageM(cmdline["right"], cv.CV_LOAD_IMAGE_UNCHANGED)

    if cropsize[2] < 0:
        cropsize = [0,0,left.cols,left.rows]


    velx = cv.CreateMat((left.rows-bsize[1])/shiftsize[1],(left.cols-bsize[0])/shiftsize[0], cv.CV_32FC1)
    vely = cv.CreateMat((left.rows-bsize[1])/shiftsize[1],(left.cols-bsize[0])/shiftsize[0], cv.CV_32FC1)

#    velx = cv.CreateMat(left.rows,left.cols, cv.CV_32FC1)
#    vely = cv.CreateMat(left.rows,left.cols, cv.CV_32FC1)

    cv.Zero(velx)
    cv.Zero(vely)
    cv.CalcOpticalFlowBM(left,right,bsize, shiftsize, maxrange, False, velx, vely)
#    cv.CalcOpticalFlowHS(left,right,False, velx, vely, 1.0, (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 10, 0.01))
#    cv.CalcOpticalFlowLK(left,right,bsize, velx, vely)


    if filtersize > 0:
        (tmp_velx, tmp_vely) = filterVelocityField(velx,vely, filtersize)
        velx = tmp_velx
        vely = tmp_vely



    intermediate_frames = []
    flow_frames = []
    for d in range (0, steps+1):
        print "Frame: %d"%d
        dst = cv.CreateMat(left.rows, left.cols, cv.CV_8UC1)
        output = cv.CloneMat(left_orig)

        cv.AddWeighted(left_orig, 1.0 - (float(d)/float(steps)), right_orig, (float(d)/float(steps)), 0.0, output) 

        cv.Zero(dst)
        (new_velx, new_vely) = createVelocityField(velx,vely,float(d)/steps)


        for x in range (0, new_velx.cols):
            for y in range (0, new_velx.rows):
                

                xpos = x*shiftsize[0] + bsize[0]/2        
                ypos = y*shiftsize[1] + bsize[1]/2        
                cv.Circle(dst, (xpos, ypos), 2, (255,255,255), 1)
                xtpos = xpos + new_velx[y,x]
                ytpos = ypos + new_vely[y,x]
                xfinalPos = xpos + velx[y,x]
                yfinalPos = ypos + vely[y,x]

                try:
                    cv.Line(dst, (int(xpos), int(ypos)), (int(xtpos), int(ytpos)), (255,0,0),1)
                    if xpos - bsize[0]/2 > 0 and xpos + bsize[0]/2 < left_orig.cols and \
                       ypos- bsize[1]/2 > 0 and ypos + bsize[1]/2 < left_orig.rows and \
                       xfinalPos - bsize[0]/2 > 0 and xfinalPos + bsize[0]/2 < right_orig.cols and \
                       yfinalPos- bsize[1]/2 > 0 and yfinalPos + bsize[1]/2 < right_orig.rows:
                        tmpMatLeft = cv.GetSubRect(left_orig, (xpos - bsize[0]/2, ypos- bsize[1]/2,bsize[0],bsize[1]))    
                        tmpMatRight = cv.GetSubRect(right_orig, (xfinalPos - bsize[0]/2, yfinalPos- bsize[1]/2,bsize[0],bsize[1]))    
                        tmpOutput = cv.GetSubRect(output, (int(xtpos) - bsize[0]/2, int(ytpos)- bsize[1]/2,bsize[0],bsize[1]))        
                        cv.AddWeighted(tmpMatLeft, 1.0 - (float(d)/float(steps)), tmpMatRight, (float(d)/float(steps)), 0.0, tmpOutput) 
                except:
                    pass # Invalid position


        flow_frames.append(dst)
        print str(cropsize)
        cropped_out = cv.CloneMat(cv.GetSubRect(output,(cropsize[0],cropsize[1],cropsize[2],cropsize[3])))
        intermediate_frames.append(cropped_out)
        cv.SaveImage("intermediate_%04d.bmp"%d, output)
        cv.SaveImage("intermediate_%04d.bmp"%(2*steps - d ), output)



    running=True
    direction=1
    pos = 0
    while running:
        pos = pos + direction

        if pos >= len(intermediate_frames)-1 or pos <= 0:
            direction = direction * (-1)

        cv.ShowImage("Output", intermediate_frames[pos])
        cv.ShowImage("Flow", flow_frames[pos])
        cv.WaitKey(20)
