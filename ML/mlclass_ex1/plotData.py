from matplotlib.pyplot import * 

def plotData(x, y):
  '''TODO the comments below are all Octave ML originals '''
  #PLOTDATA Plots the data points x and y into a new figure 
  #   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
  #   population and profit.
  
  fig = figure()
  # ====================== YOUR CODE HERE ======================
  # Instructions: Plot the training data into a figure using the 
  #               "figure" and "plot" commands. Set the axes labels using
  #               the "xlabel" and "ylabel" commands. Assume the 
  #               population and revenue data have been passed in
  #               as the x and y arguments of this function.
  #
  # Hint: You can use the 'rx' option with plot to have the markers
  #       appear as red crosses. Furthermore, you can make the
  #       markers larger by using plot(..., 'rx', markersize=10);

  ax = fig.add_subplot(1,1,1) # 1,1,1은 꼭 붙여야 한다. 아니면 NonType
  ax.scatter(x, y, marker = 'x', color = 'red', ) # markersize = 10는 없다.xlabel = 'x', ylabel = 'y'
  # 아니 씨발 plot에는 markersize property가 있는데 scatter에는 없다? s다? 이런 욕나오네.
  # fig가 아니라 ax에 scatter를 붙일 수 있다.

  # ============================================================
  return fig