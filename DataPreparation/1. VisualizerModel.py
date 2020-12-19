import pandas as pd
Signals = pd.read_csv('Signals/071309_w_21-BL1-081_bio.csv',sep = 	'	').iloc[:,:].values
Signals_Filtered = pd.read_csv('Signals/071309_w_21-BL1-081_bio.csv',sep = 	'	').iloc[:,:].values

''' Visualize GSR '''
X = Signals[:,0]
Y = Signals[:,1]
Y_Filtered = Signals_Filtered[:,1]
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker   
plt.title('Signals Visualisation')
plt.xlabel('Time')
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%d µs'))
plt.ylabel('GSR Signal')
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
plt.plot(X,Y)
plt.plot(X,Y_Filtered)
plt.show()  


''' Visualize ECG '''
X = Signals[:,0]
Y = Signals[:,2]
Y_Filtered = Signals_Filtered[:,2]
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker   
plt.title('Signals Visualisation')
plt.xlabel('Time')
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%d µs'))
plt.ylabel('ECG Signal')
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
plt.plot(X,Y)
plt.plot(X,Y_Filtered)
plt.show()  


''' Visualize EMG_Trapezius '''
X = Signals[:,0]
Y = Signals[:,3]
Y_Filtered = Signals_Filtered[:,3]
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker   
plt.title('Signals Visualisation')
plt.xlabel('Time')
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%d µs'))
plt.ylabel('EMG Trapezius Signal')
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
plt.plot(X,Y)
plt.plot(X,Y_Filtered)
plt.show()  


''' Visualize EMG_Corrugator '''
X = Signals[:,0]
Y = Signals[:,4]
Y_Filtered = Signals_Filtered[:,4]
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker   
plt.title('Signals Visualisation')
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%d µs'))
plt.ylabel('EMG Corrugator Signal')
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
plt.plot(X,Y)
plt.plot(X,Y_Filtered)
plt.show()  


''' Visualize EMG_Zygomaticus '''
X = Signals[:,0]
Y = Signals[:,5]
Y_Filtered = Signals_Filtered[:,5]
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker   
plt.title('Signals Visualisation')
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%d µs'))
plt.ylabel('EMG Zygomaticus Signal')
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
plt.plot(X,Y)
plt.plot(X,Y_Filtered)
plt.show()  
