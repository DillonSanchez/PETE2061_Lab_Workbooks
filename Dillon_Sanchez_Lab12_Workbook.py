#Modify the y-axis limits so that the maximum Cum. production gets displayed for each well
#
##Write code that will query the database and plot all gas wells
#dcaDF = pd.read_sql_query("SELECT wellID FROM DCAparams WHERE fluid='gas';", conn)
# 
##Write code that will query the database for the well with exponential decline. Print the corresponding wellID
#dcaDF = pd.read_sql_query("SELECT wellID FROM DCAparams WHERE b=0;", conn) 
#Can you stack plot all oil wells and stack plot all gas wells in the field?


import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

import pandas as pd
import sqlite3

#create a database named "DCA.db" in the folder where this code is located
conn = sqlite3.connect("DCA.db")  #It will only connect to the DB if it already exists

#create data table to store summary info about each case/well
cur = conn.cursor()

#Custom Plot parameters
titleFontSize = 18
axisLabelFontSize = 15
axisNumFontSize = 13


#1

for wellID in range(1,18):

    prodDF = pd.read_sql_query(f"SELECT time,rate,Cum FROM Rates WHERE wellID={wellID};", conn)    
    dcaDF = pd.read_sql_query("SELECT * FROM DCAparams;", conn) #this will grab everything in DCAparams table  
    
    # Primary and Secondary Y-axes
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(prodDF['time'], prodDF['rate'], color="green", ls='None', marker='o', markersize=5,)
    ax2.plot(prodDF['time'], prodDF['Cum']/1000, 'b-')

    ax1.set_xlabel('Time, Months')
    ax1.set_ylabel('Production Rate, bopm', color='g')
    ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')

    plt.show()


#2

prodDF.drop(["rate","Cum"], axis=1, inplace=True)
dcaDF=pd.read_sql_query("Select wellID FROM DCAparams WHERE fluid='gas';", conn)

for i in dcaDF['wellID']:
    prodDF['Well'+str(i)]=pd.read_sql_query(f"SELECT rate FROM Rates WHERE wellID={i};", conn)

production= prodDF.iloc[:,1:].values
time= prodDF['time'].values

labels=prodDF.columns
labels=list(labels[1:])
#print(labels)
fig, ax = plt.subplots()
ax.stackplot(time, np.transpose(production), labels=labels)
ax.legend(loc='upper right')
plt.title('Stacked Field Gas Production')
plt.show()


#3

oilRatesDF=pd.DataFrame(prodDF['time'])
dcaDF=pd.read_sql_query("Select wellID FROM DCAparams WHERE fluid='oil';", conn)

for i in dcaDF['wellID']:
    oilRatesDF['Well'+str(i)]=pd.read_sql_query(f"SELECT rate FROM Rates WHERE wellID={i};", conn)

production= oilRatesDF.iloc[:,1:].values
time= oilRatesDF['time'].values

labels=oilRatesDF.columns
labels=list(labels[1:])
#print(labels)
fig, ax = plt.subplots()
ax.stackplot(time, np.transpose(production), labels=labels)
ax.legend(loc='upper right')
plt.title('Stacked Field Oil Production')
plt.show()


#4

#stacked bar graph
N = 6
ind = np.arange(1,N+1)    # the x locations for the groups
months = ['Jan','Feb','Mar','Apr','May','Jun']
result=np.zeros(len(months))
labels=[]
loc_plts=[]
width = 0.5       # the width of the bars

cumDF=pd.DataFrame(prodDF['time'])
dcaDF=pd.read_sql_query("Select wellID FROM DCAparams WHERE fluid='gas';", conn)

for i in dcaDF['wellID']:
    cumDF['Well'+str(i)]=pd.read_sql_query(f"SELECT Cum FROM Rates WHERE wellID={i};", conn)

j=1
for i in dcaDF['WellID']:
    p1 = plt.bar(cumDF['time'][0:N], cumDF['Well'+str(i)][0:N]/1000, width, bottom=result)
    labels.append('Well'+str(i))
    loc_plts.append(p1)
    plt.ylabel('Gas Production, Mbbls')
    plt.title('Cumulative Gas Feild Production')
    plt.xticks(ind, months, fontweight='bold')
    j+=1
    split=cumDF.iloc[0:6,1:j].values
    result=np.sum(split,axis=1)/1000

plt.legend(loc_plts,labels)
plt.show(loc_plts)


#5

#stacked bar graph
N = 6
ind = np.arange(1,N+1)    # the x locations for the groups
months = ['Jan','Feb','Mar','Apr','May','Jun']
result=np.zeros(len(months))
labels=[]
loc_plts=[]
width = 0.5       # the width of the bars

cumDF=pd.DataFrame(prodDF['time'])
dcaDF=pd.read_sql_query("Select wellID FROM DCAparams WHERE fluid='oil';", conn)

for i in dcaDF['wellID']:
    cumDF['Well'+str(i)]=pd.read_sql_query(f"SELECT Cum FROM Rates WHERE wellID={i};", conn)

j=1
for i in dcaDF['WellID']:
    p1 = plt.bar(cumDF['time'][0:N], cumDF['Well'+str(i)][0:N]/1000, width, bottom=result)
    labels.append('Well'+str(i))
    loc_plts.append(p1)
    plt.ylabel('Oil Production, Mbbls')
    plt.title('Cumulative Oil Feild Production')
    plt.xticks(ind, months, fontweight='bold')
    j+=1
    split=cumDF.iloc[0:6,1:j].values
    result=np.sum(split,axis=1)/1000

plt.legend(loc_plts,labels)
loc_plts=plt.figure(figsize=(36,20), dpi=100)
#plt.show(loc_plts)


#6

#two approaches to load a LAS input file
data1 = np.loadtxt("volve_logs/15_9-F-1B_INPUT.LAS", skiprows=69)

#Load and prepare the data
DZ1,rho1=data1[:,0], data1[:,16]

#clean data where negative density
DZ1=DZ1[np.where(rho1>0)]
rho1=rho1[np.where(rho1>0)]

titleFontSize = 22
fontSize = 20
#Plotting multiple well log tracks on one graph
fig = plt.figure(figsize=(36,20),dpi=100)
fig.tight_layout(pad=1, w_pad=4, h_pad=2)

plt.subplot(1, 6, 1)
plt.grid(axis='both')
plt.plot(rho1,DZ1, color='red')
#plt.plot(rho*1.1,DZ, color='blue')
plt.title('Density vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1,DT1=data1[:,0],data1[:,8]
DZ1=DZ1[np.where(DT1>0)]
DT1=DT1[np.where(DT1>0)]

plt.subplot(1, 6, 2)
plt.grid(axis='both')
plt.plot(DT1,DZ1, color='green')
plt.title('DT vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('DT, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1,DTS1=data1[:,0],data1[:,9]
DZ1=DZ1[np.where(DTS1>0)]
DTS1=DTS1[np.where(DTS1>0)]

plt.subplot(1, 6, 3)
plt.grid(axis='both')
plt.plot(DTS1,DZ1, color='blue')
plt.title('DTS vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('DTS, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1,GR1=data1[:,0],data1[:,10]
DZ1=DZ1[np.where(GR1>0)]
GR1=GR1[np.where(GR1>0)]

plt.subplot(1, 6, 4)
plt.grid(axis='both')
plt.plot(GR1,DZ1, color='black')
plt.title('GR vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('GR, API', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1,NPHI1=data1[:,0],data1[:,12]
DZ1=DZ1[np.where(NPHI1>0)]
NPHI1=NPHI1[np.where(NPHI1>0)]

plt.subplot(1, 6, 5)
plt.grid(axis='both')
plt.plot(NPHI1,DZ1, color='brown')
plt.title('NPHI vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('NPHI, v/v', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1,CALI1=data1[:,0],data1[:,10]
DZ1=DZ1[np.where(CALI1>0)]
CALI1=CALI1[np.where(CALI1>0)]

plt.subplot(1, 6, 6)
plt.grid(axis='both')
plt.plot(CALI1,DZ1, color='grey')
plt.title('Caliper vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Caliper, inch', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

fig.savefig('Well_15_9-F-4.png', dpi=600)



data2 = np.loadtxt("volve_logs/15_9-F-1B_INPUT.LAS", skiprows=69)

#Load and prepare the data
DZ2,rho2=data2[:,0], data2[:,16]

#clean data where negative density
DZ2=DZ2[np.where(rho2>0)]
rho2=rho2[np.where(rho2>0)]

titleFontSize = 22
fontSize = 20
#Plotting multiple well log tracks on one graph
fig = plt.figure(figsize=(36,20),dpi=100)
fig.tight_layout(pad=1, w_pad=4, h_pad=2)

plt.subplot(1, 6, 1)
plt.grid(axis='both')
plt.plot(rho2,DZ2, color='red')
#plt.plot(rho*1.1,DZ, color='blue')
plt.title('Density vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ2,DT2=data2[:,0],data2[:,8]
DZ2=DZ2[np.where(DT2>0)]
DT2=DT2[np.where(DT2>0)]

plt.subplot(1, 6, 2)
plt.grid(axis='both')
plt.plot(DT2,DZ2, color='green')
plt.title('DT vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('DT, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ2,DTS2=data2[:,0],data2[:,9]
DZ2=DZ2[np.where(DTS2>0)]
DTS2=DTS2[np.where(DTS2>0)]

plt.subplot(1, 6, 3)
plt.grid(axis='both')
plt.plot(DTS2,DZ2, color='blue')
plt.title('DTS vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('DTS, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ2,GR2=data1[:,0],data2[:,10]
DZ2=DZ2[np.where(GR2>0)]
GR2=GR2[np.where(GR2>0)]

plt.subplot(1, 6, 4)
plt.grid(axis='both')
plt.plot(GR2,DZ2, color='black')
plt.title('GR vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('GR, API', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ2,NPHI2=data2[:,0],data2[:,12]
DZ2=DZ2[np.where(NPHI2>0)]
NPHI2=NPHI2[np.where(NPHI2>0)]

plt.subplot(1, 6, 5)
plt.grid(axis='both')
plt.plot(NPHI2,DZ2, color='brown')
plt.title('NPHI vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('NPHI, v/v', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ2,CALI2=data2[:,0],data2[:,10]
DZ2=DZ2[np.where(CALI2>0)]
CALI2=CALI2[np.where(CALI2>0)]

plt.subplot(1, 6, 6)
plt.grid(axis='both')
plt.plot(CALI2,DZ2, color='grey')
plt.title('Caliper vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Caliper, inch', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

fig.savefig('Well_15_9-F-1B.png', dpi=600)



data3 = np.loadtxt("volve_logs/15_9-F-1B_INPUT.LAS", skiprows=69)

#Load and prepare the data
DZ3,rho3=data3[:,0], data3[:,16]

#clean data where negative density
DZ3=DZ3[np.where(rho3>0)]
rho3=rho3[np.where(rho3>0)]

titleFontSize = 22
fontSize = 20
#Plotting multiple well log tracks on one graph
fig = plt.figure(figsize=(36,20),dpi=100)
fig.tight_layout(pad=1, w_pad=4, h_pad=2)

plt.subplot(1, 6, 1)
plt.grid(axis='both')
plt.plot(rho3,DZ3, color='red')
#plt.plot(rho*1.1,DZ, color='blue')
plt.title('Density vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ3,DT3=data3[:,0],data3[:,8]
DZ3=DZ3[np.where(DT3>0)]
DT3=DT3[np.where(DT3>0)]

plt.subplot(1, 6, 2)
plt.grid(axis='both')
plt.plot(DT3,DZ3, color='green')
plt.title('DT vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('DT, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ3,DTS3=data3[:,0],data3[:,9]
DZ3=DZ3[np.where(DTS3>0)]
DTS3=DTS3[np.where(DTS3>0)]

plt.subplot(1, 6, 3)
plt.grid(axis='both')
plt.plot(DTS3,DZ3, color='blue')
plt.title('DTS vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('DTS, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ3,GR3=data3[:,0],data3[:,10]
DZ3=DZ3[np.where(GR3>0)]
GR3=GR3[np.where(GR3>0)]

plt.subplot(1, 6, 4)
plt.grid(axis='both')
plt.plot(GR3,DZ3, color='black')
plt.title('GR vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('GR, API', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ3,NPHI3=data3[:,0],data3[:,12]
DZ3=DZ3[np.where(NPHI3>0)]
NPHI3=NPHI3[np.where(NPHI3>0)]

plt.subplot(1, 6, 5)
plt.grid(axis='both')
plt.plot(NPHI3,DZ3, color='brown')
plt.title('NPHI vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('NPHI, v/v', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ3,CALI3=data3[:,0],data3[:,10]
DZ3=DZ3[np.where(CALI3>0)]
CALI3=CALI3[np.where(CALI3>0)]

plt.subplot(1, 6, 6)
plt.grid(axis='both')
plt.plot(CALI3,DZ3, color='grey')
plt.title('Caliper vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Caliper, inch', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()



fig.savefig('Well_15_9-F-14.png', dpi=600)





conn.close()





##Syntax to add new columns to a table
#cur.execute("ALTER TABLE Rates ADD rateID INTEGER;")
#conn.commit()

#Syntax to delete a table
#cur.execute("DROP TABLE DCAparams;")
#cur.execute("DROP TABLE Rates;")
#conn.commit()
