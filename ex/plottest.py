import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('click on points')

line1, = ax.plot(np.random.rand(100), 'bo', picker=5)  # 5 points tolerance
line2, = ax.plot(np.random.rand(100), 'ro', picker=5)  # 5 points tolerance
print(line1)
print(line2)

def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    print(thisline)
    print('onpick points:', zip(xdata[ind], ydata[ind]))

fig.canvas.mpl_connect('pick_event', onpick)
plt.show()
