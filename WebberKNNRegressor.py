import math,random,time
def recommend(URL,testScale,logList,criterionList,predictorList,kList,weightList):
    preprocessStart = time.time()
    users = []
    for user in open(URL,'r').readlines():
        userRatings = []
        ratingElement = user.split(',')
        for element in ratingElement[:-2]:
            if element is not '':
                userRatings.append(int(element))
            else:
                userRatings.append(None)
        for element in ratingElement[-2:]:
            userRatings.append(float(element))
        users.append(userRatings)
    rowNum = len(users)
    columnNum = len(users[0])
    corrMatrix = [[None]*columnNum]*rowNum
    overlapNumMatrix = [[None]*columnNum]*rowNum
    overlapSDMatrix = [[None]*columnNum]*rowNum
    overlapMeanMatrix = [[None]*columnNum]*rowNum
    
    for cursorIndex in range(1,rowNum):
        for anotherIndex in range(cursorIndex+1,rowNum):
            
            cursorOverlapList = []
            anotherOverlapList = []
            for movieIndex in range(1,columnNum):
                if not users[cursorIndex][movieIndex] and not users[anotherIndex][movieIndex]:
                    cursorOverlapList.append(users[cursorIndex][movieIndex])
                    anotherOverlapList.append(users[anotherIndex][movieIndex])
            
            if len(cursorOverlapList) > 1:
                overlapNumMatrix[cursorIndex][anotherIndex] = len(cursorOverlapList)
                overlapNumMatrix[anotherIndex][cursorIndex] = len(anotherOverlapList)
                
                cursorOverlapMean = float(sum(cursorOverlapList)) / len(cursorOverlapList)
                anotherOverlapMean = float(sum(anotherOverlapList)) / len (anotherOverlapList)
                overlapMeanMatrix[cursorIndex][anotherIndex] = cursorOverlapMean
                overlapMeanMatrix[anotherIndex][cursorIndex] = anotherOverlapMean               
                
                cursorOverlapSD = calculateSD(cursorOverlapList,cursorOverlapMean)
                anotherOverlapSD = calculateSD(anotherOverlapList,anotherOverlapMean)               
                overlapSDMatrix[cursorIndex][anotherIndex] = cursorOverlapSD
                overlapSDMatrix[anotherIndex][cursorIndex] = anotherOverlapSD
                
                errorProductSum = 0.0
                for overlapIndex in len(cursorOverlapList):
                    errorProductSum += (cursorOverlapList[overlapIndex] - cursorOverlapMean) * (anotherOverlapList[overlapIndex] - anotherOverlapMean)
                try:
                    Corr = errorProductSum / len(cursorOverlapList) / cursorOverlapSD / anotherOverlapSD
                except:
                    if cursorOverlapSD == anotherOverlapSD == 0:
                        Corr = 1
                corrMatrix[cursorIndex][anotherIndex] = Corr
                corrMatrix[anotherIndex][cursorIndex] = Corr
    preprocessTime = time.time() - preprocessStart
    showTime('Preprocess',preprocessTime)

    indecies = set(range(1,rowNum))
    trainingSet = set([])
    for i in indecies:
        if random.random() > testScale:
            trainingSet.add(i)
    testSet = indecies - trainingSet
    
    for log in logList:
        for criterion in criterionList:
            for predictor in predictorList:
                for k in kList:
                    for weight in weightList:
                        applyModel(trainingSet,users,corrMatrix,overlapNumMatrix,log,criterion,predictor,k,weight,overlapMeanMatrix,overlapSDMatrix)
            pass
            
def applyModel(appliedSet,users,corrMatrix,overlapNumMatrix,log,criterion,predictor,k,weightFLG,overlapMeanMatrix,overlapSDMatrix):
    errorList = []   
    for cursorIndex in appliedSet:
        kNN = []
        kNNCorr = []
        for anotherIndex in range(1,len(corrMatrix)):
            if corrMatrix[cursorIndex][anotherIndex] is not None:
                Corr = 0.0
                if overlapNumMatrix[cursorIndex][anotherIndex] < criterion:
                    Corr = 0
                else:
                    Corr = corrMatrix[cursorIndex][anotherIndex] * math.log(overlapNumMatrix[cursorIndex][anotherIndex],criterion) if log else corrMatrix[cursorIndex][anotherIndex]
                index = binaryInsert(0,len(kNNCorr),kNNCorr,Corr)
                kNN.insert(index,users[anotherIndex])
                kNNCorr.insert(index,Corr)
        errorList +=  predictor(users[cursorIndex],kNN,kNNCorr,k,weightFLG,overlapMeanMatrix,overlapSDMatrix)
    return {'log':log,'criterion':criterion,'predictor':predictor.__name__[:-9],'k':k,'weight':weightFLG,'errorNum':len(errorList),'minError':min(errorList),'maxError':max(errorList),'meanError':sum(errorList)/len(errorList)}    
          
        
def meanPredictor(user,kNN,kNNCorr,k,weightFLG,overlapMeanMatrix,overlapSDMatrix):
    errorList = []
	for movieIndex in range(1,len(user)):
		kCount = 0
		valueSum = 0.0
		weightSum = 0.0
		if not user[movieIndex]:
			for kIndex in range(len(kNN)):
				if kCount >= k and kNNCorr[kIndex]<=0:
						break
				weight = kNNCorr[kIndex] if weightFLG else 1
				if not kNN[kIndex][movieIndex]:
					valueSum += kNN[kIndex][movieIndex] * weight
					weightSum += weight
					kCount += 1
		try:
			errorList.append(valueSum / weightSum - user[movieIndex])
		except:
			if not kCount:
				errorList.append(valueSum / kCount - user[movieIndex])
    return errorList

def differenPredictor(user,kNN,kNNCorr,k,weightFLG,overlapMeanMatrix,overlapSDMatrix):
    errorList = []
	for movieIndex in range(1,len(user)):
		kCount = 0
		differSum = 0.0
		weightSum = 0.0
		if not user[movieIndex]:
			for kIndex in range(len(kNN)):
				if kCount >= k and kNNCorr[kIndex]<=0:
						break
				weight = kNNCorr[kIndex] if weightFLG else 1
				if not kNN[kIndex][movieIndex]:
					differSum += (kNN[kIndex][movieIndex] - kNN[kIndex][-2]) * weight
					weightSum += weight
					kCount += 1
		try:
			errorList.append(differSum / weightSum + user[-2] - user[movieIndex])
		except:
			if not kCount:
				errorList.append(differSum / kCount + user[-2] - user[movieIndex])
    return errorList

def zScorePredictor(user,kNN,kNNCorr,k,weightFLG,overlapMeanMatrix,overlapSDMatrix):
    errorList = []
	for movieIndex in range(1,len(user)):
		kCount = 0
		zScoreSum = 0.0
		weightSum = 0.0
		if not user[movieIndex]:
			for kIndex in range(len(kNN)):
				if kCount >= k and kNNCorr[kIndex]<=0:
						break
				weight = kNNCorr[kIndex] if weightFLG else 1
				if not kNN[kIndex][movieIndex]:
					zScoreSum += ((kNN[kIndex][movieIndex] - kNN[kIndex][-2]) * kNNCorr[kIndex] / kNN[kIndex][-1]) * weight
					weightSum += weight
					kCount += 1
		try:
			errorList.append(zScoreSum * user[-1] / weightSum + user[-2] - user[movieIndex])
		except:
			if not kCount:
				errorList.append(zScoreSum * user[-1] / kCount + user[-2] - user[movieIndex])
    return errorList

def webberPredictor(user,kNN,kNNCorr,k,weightFLG,overlapMeanMatrix,overlapSDMatrix):
    errorList = []
    cursorIndex = user[0]-1
	for movieIndex in range(1,len(user)):
		kCount = 0
		valueSum = 0.0
		weightSum = 0.0
		if not user[movieIndex]:
			for kIndex in range(len(kNN)):
				if kCount >= k and kNNCorr[kIndex]<=0:
						break
				weight = kNNCorr[kIndex] if weightFLG else 1
				if not kNN[kIndex][movieIndex]:
					anotherIndex = kNN[kIndex][0]-1
					valueSum += ((kNN[kIndex][movieIndex] - overlapMeanMatrix[anotherIndex][cursorIndex]) / overlapSDMatrix[anotherIndex][cursorIndex] * overlapSDMatrix[cursorIndex][anotherIndex] + overlapMeanMatrix[cursorIndex][anotherIndex]) * weight
					weightSum += weight
					kCount += 1
		try:
			errorList.append(valueSum / weightSum - user[movieIndex])
		except:
			if not kCount:
				errorList.append(valueSum / kCount - user[movieIndex])
    return errorList
    
def binaryInsert(minIndex,maxIndex,list,element):
    if maxIndex - minIndex < 2:
        if len(list) != 0 and element < list[minIndex]:
            return maxIndex
        else:
            return minIndex
    midIndex = (minIndex + maxIndex) / 2
    if element < list[midIndex]:
        return binaryInsert(midIndex,maxIndex,list,element)
    else:
        return binaryInsert(minIndex,midIndex,list,element)
        
def calculateSD(list,mean):
    if len(list) != 0:
        sumOfSquareError = 0.0
        for i in list:
            sumOfSquareError += math.square(i - mean)
        return sumOfSquareError / len(list)

def showTime(processTime,label='Time'):
    hour =  int(processTime) / 3600
    minute = int(processTime - (hour * 3600)) / 60
    second = processTime - (hour * 3600) - (minute * 60)
    print label+': ',hour,'h ',minute,'m ',second,'s'