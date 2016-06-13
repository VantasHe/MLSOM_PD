import numpy as np

filename = 'iris.data'

train_f = open(filename, 'r')

new_data = []
count = [1]
for data in train_f:
    element = data.split(',')
    #new_element = count + element[0:4]
    #new_data.append(new_element)
    new_data.append(element[0:4])
    count[0] = count[0] + 1
    
for dataPrint in new_data:
    print(dataPrint, sep=' ')    

#====Initializing====#

traCount = 150
totalNeuro = 3
neuroOfSOM = []

for neuro in range(0, totalNeuro, 1):
    neuroOfSOM = np.random.rand(4,1)
print(neuroOfSOM)
for i in range(0, traCount, 1):
    aMatrix = np.mat(new_data[i], 'f4')
    for j in range(0, totalNeuro, 1):
        sqrt_weight = np.power(neuroOfSOM[j], 2)
        sqrt_aMatrix = np.power(aMatrix,2)
        result = abs(sqrt_aMatrix - sqrt_weight)
        print(result)
        result = np.array(result).flatten().tolist()
        print(sum(result))