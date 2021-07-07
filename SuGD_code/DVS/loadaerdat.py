# Source:
# https://sourceforge.net/p/jaer/code/HEAD/tree/scripts/python/jAER_utils/loadaerdat.py
import struct
import os
import numpy as np

def loadaerdata(datafile, length=0, debug=1):
    """
    load AER data file and parse these properties of AE events:
    - timestamps (in us),
    - x,y-position [0..127]
    - polarity (0/1)
    @param datafile - path to the file to read
    @param length - how many bytes(B) should be read; default 0=whole file
    @param version - which file format version is used: "aedat" = v2, "dat" = v1 (old)
    @param debug - 0 = silent, 1 (default) = print summary, >=2 = print all debug
    @param camera='DVS128' or 'DAVIS240'
    @return (ts, xpos, ypos, pol) 4-tuple of lists containing data of all events;
    """
    # constants
    aeLen = 8  # 1 AE event takes 8 bytes
    headerLen = 28
    td = 0.000001  # timestep is 1us
    xmask = 0x00001FFF
    xshift = 17
    ymask = 0x00001FFF
    yshift = 2
    pmask = 0x1
    pshift = 1

    print(datafile)
    aerdatafh = open(datafile, 'rb')
    p = 0  # pointer, position on bytes
    statinfo = os.stat(datafile)
    if length == 0:
        length = statinfo.st_size
    print("file size", length)

    # header
    lt = aerdatafh.readline()
    # print(str(lt[0]))
    while lt and str(lt)[2] == '#':
        p += len(lt)
        if debug >= 2:
            print(str(lt))
        lt = aerdatafh.readline()
        continue

    # variables to parse
    timestamps = []
    xaddr = []
    yaddr = []
    pol = []

    # read data-part of file
    print(xmask, xshift, ymask, yshift, pmask, pshift)
    while p < length:
        aerdatafh.seek(p)
        header = aerdatafh.read(headerLen)
        p += headerLen

        eventType, eventSource, eventSize, eventTSOffset, eventTSOverflow, eventCapacity, eventNumber, eventValid = struct.unpack('<HHIIIIII', header)

        # print(eventNumber)

        for i in range(eventNumber):
            aerdatafh.seek(p)
            s = aerdatafh.read(aeLen)
            p += aeLen

            addr, ts = struct.unpack('<II', s)

            x_addr = (addr >> xshift) & xmask
            y_addr = (addr >> yshift) & ymask
            a_pol = (addr >> pshift) & pmask

            if debug >= 3:
                print("ts->", ts)  # ok
                print("x-> ", x_addr)
                print("y-> ", y_addr)
                print("pol->", a_pol)

            timestamps.append(ts)
            xaddr.append(x_addr)
            yaddr.append(y_addr)
            pol.append(a_pol)

    if debug > 0:
        try:
            print("read %i (~ %.2fM) AE events, duration= %.2fs" % (
            len(timestamps), len(timestamps) / float(10 ** 6), (timestamps[-1] - timestamps[0]) * td))
            n = 10
            print("showing first %i:" % (n))
            print("timestamps: %s \nX-addr: %s\nY-addr: %s\npolarity: %s" % (
            timestamps[0:n], xaddr[0:n], yaddr[0:n], pol[0:n]))
        except:
            print("failed to print statistics")

    return np.asarray(timestamps), np.asarray(xaddr), np.asarray(yaddr), np.asarray(pol)


