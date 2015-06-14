from numpy import *
from matplotlib.pyplot import *

def drawLine(p1, p2, *args, **kwargs):
    #DRAWLINE Draws a line from point p1 to point p2
    #   DRAWLINE(p1, p2) Draws a line from point p1 to point p2 and holds the
    #   current figure

    plot(array([p1[0], p2[0]]), array([p1[1], p2[1]]), *args, **kwargs)
