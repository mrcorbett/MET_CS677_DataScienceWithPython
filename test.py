nUnique = 786

updateFreq = 100
updateIndex = updateFreq
updateNext = updateFreq

unique = range(0, 786)
print(len(unique))
print('updateIndex={} updateNext={}'.format(updateIndex, updateNext))

for index, value in enumerate(unique):
    if index == updateIndex:
        updateIndex += updateFreq
        updateNext += updateFreq
        #print('updateIndex={} updateNext={}'.format(updateIndex, updateNext))
        print(updateIndex//updateFreq)
print('\tcomplete')

