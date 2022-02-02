from twoPairCNS import *



start=1
endPast1=5+1
paisAll=[]

# calcTimeStart=datetime.now()

for i in range(start,endPast1-1):

    for j in range(i,endPast1):
        if math.gcd(i,j)>1:
            continue
        else:
            paisAll.append([i,j])

for pairTmp in paisAll:
    calcTimeStart = datetime.now()
    iTmp,jTmp=pairTmp
    dtPack1=dataPack(iTmp,jTmp)
    dtPack2=dataPack(jTmp,iTmp)
    calcChernNumberAndPlot(dtPack1,dtPack2)
    calcTimeEnd = datetime.now()
    print("calculation time: ", calcTimeEnd - calcTimeStart)

# calcTimeEnd=datetime.now()
# print("calculation time: ",calcTimeEnd-calcTimeStart)