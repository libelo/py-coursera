from numpy import *
from matplotlib.pyplot import *

def displayData(X, example_width=None):
#DISPLAYDATA Display 2D data in a nice grid
#   display_array = DISPLAYDATA(X, example_width) displays 2D data
#   stored in X in a nice grid in the current figure. It returns
#   the displayed array.

    # Set example_width automatically if not passed in
    if example_width is None:
        example_width = int(sqrt(size(X, 1))+0.5)

    # Compute rows, cols
    m, n = shape(X)
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(sqrt(m))
    display_cols = (m + display_rows - 1) / display_rows

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = - ones((pad + display_rows * (example_height + pad),
                            pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
    	for i in range(display_cols):
            if curr_ex >= m:
    			break
            # Copy the patch

            # Get the max value of the patch
            max_val = max(abs(X[curr_ex, :]))
            pos_x = pad + i * (example_width + pad)
            pos_y = pad + j * (example_height + pad)
            display_array[pos_y : pos_y + example_height, pos_x : pos_x + example_width] = \
                reshape(X[curr_ex, :], (example_height, example_width), order='F') / max_val
            curr_ex += 1
    	if curr_ex >= m:
    		break

    # Display Image
    imshow(display_array, interpolation='none', cmap=cm.gray)

    # Do not show axis
    axis('off')

    return display_array
