import csv
import math

def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    #print("vivek")
    dataset = list(lines)
    #print("abc")
    for i in range(1,len(dataset)):
        #del dataset[i][0]
        del dataset[i][0]
        del dataset[i][2]
        del dataset[i][2]
        del dataset[i][2]
        del dataset[i][2]
        del dataset[i][2]
        del dataset[i][2]
        del dataset[i][2]
        del dataset[i][2]
        del dataset[i][2]
        dataset[i-1] = [float(x) for x in dataset[i]]
        #print(dataset[i-1])
    dataset[len(dataset)-1] = [float(x) for x in [0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    print(dataset[0])
    return dataset

def getMeasures(testSet, predictions):
    correct = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            if predictions[i] == 1:
                tp +=1
            else:
                tn +=1
            correct += 1
        if testSet[i][-1] != predictions[i]:
            if predictions[i] == 1:
                fp +=1
            else:
                fn +=1
    accuracy = correct/float(len(testSet))
    percentAccuracy = accuracy * 100.0
    return [percentAccuracy, tp, tn, fp, fn]

def mean(num):
    sumVal = sum(num)
    length = float(len(num))
    return sumVal/length

def classSeparation(dataset):
    distribute = {}
    for i in range(len(dataset)):
        setC = dataset[i]
        if (setC[-1] not in distribute):
            distribute[setC[-1]] = []
        distribute[setC[-1]].append(setC)
    return distribute

def standarddev(num):
    avg = mean(num)
    temp = float(len(num)-1)
    var = sum([pow(x-avg,2) for x in num])/temp
    sdev = math.sqrt(var)
    return sdev

def calculateProbability(x, mean, stdev):
    d = (2*math.pow(stdev,2))
    if d==0:
        d=0.000001
    power = math.pow(x-mean,2)
    exponent = math.exp(-(power/d))
    c = (math.sqrt(2*math.pi) * stdev)
    if c==0:
        c=0.000001
    prob = (1 / c) * exponent
    return prob

def summarizeSet(dataset):
    content = [(mean(attr), standarddev(attr)) for attr in zip(*dataset)]
    del content[-1]
    return content

def classProbabilities(content, inp):
    probabilities = {}
    for cVal, classSummaries in content.iteritems():
        probabilities[cVal] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inp[i]
            probabilities[cVal] *= calculateProbability(x, mean, stdev)
    return probabilities
			
def predictOverall(content, inp):
    probabilities = classProbabilities(content, inp)
    bestLabel, bestProb = None, -1
    for cVal, probab in probabilities.iteritems():
        if bestLabel is None or probab > bestProb:
            bestProb = probab
            bestLabel = cVal
    return bestLabel

def classSummarization(datasetVal):
    distribute = classSeparation(datasetVal)
    content = {}
    for classValue, val in distribute.iteritems():
        content[classValue] = summarizeSet(val)
    return content

def accumulatePredictions(content, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predictOverall(content, testSet[i])
        predictions.append(result)
    return predictions

def dataSplit(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = dataset[0:trainSize]
    copy = dataset[trainSize : len(dataset)]
    return [trainSet, copy]

def main():
    filename = 'train.csv'
    splitRatio = 0.70
    dataset = loadCsv(filename)
    #print('vivek')
    trainingSet, testSet = dataSplit(dataset, splitRatio)
    content = classSummarization(trainingSet)
    predictions = accumulatePredictions(content, testSet)
    #print("qqqq")
    accuracy, tp, tn, fp, fn = getMeasures(testSet, predictions)
    print('Accuracy: {0}%').format(accuracy)
    print ('True positive: {0}').format(tp)
    print ('True negative: {0}').format(tn)
    print ('False positive: {0}').format(fp)
    print ('Fale negative: {0}').format(fn)
    precision = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    print precision
    print recall
    fmeasure = 2*precision*recall/(precision+recall)
    print('F-measure: ')
    print fmeasure
    print('Sensitivity: ')
    print tp/float(tp+fn)

main()