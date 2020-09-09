#!/usr/bin/env python3 

fen = open('fen.csv', 'r') 
testSets = []
testSets.append(open('testset1.txt', 'w'))
testSets.append(open('testset2.txt', 'w'))
testSets.append(open('testset3.txt', 'w'))
testSets.append(open('testset4.txt', 'w'))
testSets.append(open('testset5.txt', 'w'))
testSets.append(open('testset6.txt', 'w'))
testSets.append(open('testset7.txt', 'w'))
testSets.append(open('testset8.txt', 'w'))
testSets.append(open('testset9.txt', 'w'))
testSets.append(open('testset10.txt', 'w'))
testSets.append(open('testset11.txt', 'w'))
testSets.append(open('testset12.txt', 'w'))
testSets.append(open('testset13.txt', 'w'))
testSets.append(open('testset14.txt', 'w'))
testSets.append(open('testset15.txt', 'w'))
testSets.append(open('testset16.txt', 'w'))
testSets.append(open('testset17.txt', 'w'))
testSets.append(open('testset18.txt', 'w'))
testSets.append(open('testset19.txt', 'w'))
testSets.append(open('testset20.txt', 'w'))
testSets.append(open('validationset.txt', 'w'))
count = 0
for line in fen: 
    testSets[count%len(testSets)].write(line)
    count += 1
    

fen.close()
for f in testSets:
    f.close()

