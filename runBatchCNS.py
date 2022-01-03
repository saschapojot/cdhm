from batchCNS import *



start=1
endPast1=2+1
paisAll=[]
elemsAll=range(start,endPast1)

for i in range(0,len(elemsAll)-1):
    for j in range(i+1,len(elemsAll)):
        paisAll.append([elemsAll[i],elemsAll[j]])

for pairTmp in paisAll:
    iTmp,jTmp=pairTmp
    startTimeTmp=datetime.now()
    producerTmp1=chernNumbersProducer(iTmp,jTmp)
    producerTmp1.calcChernNumberAndPlot()
    producerTmp2=chernNumbersProducer(jTmp,iTmp)
    producerTmp2.calcChernNumberAndPlot()
    endTimeTmp=datetime.now()
    print("pair  ["+str(iTmp)+", "+str(jTmp)+"] time: ",endTimeTmp-startTimeTmp)
