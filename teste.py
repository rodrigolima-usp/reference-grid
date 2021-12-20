from scipy import io
import matplotlib.pyplot as plt

mat = io.loadmat('day01.mat')

data = mat['data']

print(data[0]['left'][0]['pose'][0][0][0])#['right'][0][0][0]['image'])

#left, right = data[0]

#left, right = left[0][0], right[0][0]

#print(mat['data'])
