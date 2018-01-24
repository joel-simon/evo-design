import sys
import numpy as np
from viewer import Viewer

target = np.ones((9, 9, 9), dtype='uint8')
r = 3
target[r:-r, :, r:-r] = 0
target[:, r:-r, r:-r] = 0
target[r:-r, r:-r, :] = 0

if __name__ == '__main__':
    # assert len(sys.argv) == 3
    view = Viewer()

    grid = np.load(sys.argv[1])
    # target = np.load(sys.argv[2])

    view.startDraw()
    for index, value in np.ndenumerate(grid):
        if value:
            view.drawCube(index,color=(.5, .5, .5, 1.0))

    for index, value in np.ndenumerate(target):
        index = list(index)
        index[0] += grid.shape[0] + 2
        if value:
            view.drawCube(index, color=(.9, .1, .1, 1.0))

    view.endDraw()
    view.mainLoop()