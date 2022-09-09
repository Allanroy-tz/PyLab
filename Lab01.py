# %%
import numpy as np

# check the number of dimensions,shape,size and data type of data
data = np.arange(12).reshape(4,3)
print(data)

print("-----Create a 3-by-4 zero array-----")
data01 = np.zeros(12).reshape(3,4)

print(data01)

print("-----Create a 3-by-4 one array-----")
data02 = np.ones(12).reshape(3,4)
print(data02)

print("-----Create a 5-by-2 empty array-----")
data03 = np.empty(10).reshape(5,2)
print(data03)

print("-----Create an array array([ 1, 6, 11, 16]) using arrange-----")
data04 = np.arange(1,20,5)
print(data04)

print("-----create a 2-by-3 zero array by specifying the data type as float64-----")
data05 = np.zeros(6,np.float64)
print(data05)
print("DTpey=",data05.dtype.name)

print("-----Show the name of the data type of the array-----")
data_one = np.array([[1,2,3],[4,5,6]])
print(data_one)
print("data_one.DTpey=",data_one.dtype.name)

print("-----Change the data type of data_one to float64-----")
data_one = data_one.astype(np.float64)
print(data_one)
print("data_one.DTpey=",data_one.dtype.name)

print("-----Change the data type of float_data = np.array([1.2, 2.3, 3.5]) to int64-----")
float_data = np.array([1.2,2.3,3.5]).astype(np.int64)
print(float_data)
print("float_data.DTpey=",float_data.dtype.name)

print("-----Change the data type of str_data = np.array(['1', '2', '3']) to int64-----")
str_data = np.array(['1','2','3']).astype(np.int64)
print(str_data)
print("str_data.DTpey=",str_data.dtype.name)

print("-----Create two arrary:data1,data2-----")
data1 = np.array([[1, 2, 3], [4, 5, 6]])
data2 = np.array([[1, 2, 3], [4, 5, 6]])
print("data1:\n",data1)
print("data2:\n",data2)

print("-----(1) data1 + data2-----")
data_out = data1+data2
print("data_out:\n",data_out)

print("-----(2) data1 * data2-----")
data_out = data1*data2
print("data_out:\n",data_out)

print("-----(3) data1 - data2-----")
data_out = data1-data2
print("data_out:\n",data_out)

print("-----(4) data1 / data2-----")
data_out = data1/data2
print("data_out:\n",data_out)

print("-----Create two arrary:arr1,arr2-----")
arr1 = np.array([[0],[1],[2],[3]])
arr2 = np.array([1,2,3])
print("arr1:\n",arr1)
print("arr2:\n",arr2)

print("-----Calculate arr1+arr2-----")
arr_out = arr1+arr2
print("arr_out:\n",arr_out)

print("-----Create two arrary:data1,data2-----")
data1 = np.array([[1,2,3],[4,5,6]])
data2 = 10
print("data1:\n",data1)
print("data2:\n",data2)


print("-----(1) data1 + data2-----")
data_out = data1+data2
print("data_out:\n",data_out)

print("-----(2) data1 * data2-----")
data_out = data1*data2
print("data_out:\n",data_out)

print("-----(3) data1 - data2-----")
data_out = data1-data2
print("data_out:\n",data_out)

print("-----(4) data1 / data2-----")
data_out = data1/data2
print("data_out:\n",data_out)

print("-----Create an array as arr = np.arange(8)-----")
arr = np.arange(0,8,1)
print("arr:\n",arr)

print("----- (1) obtain the 6th element-----")
out = arr[5]
print("out:",out)

print("----- (2) obtain the 4th to 5th elements using “:” sign-----")
out = arr[3:4:1]
print("out:",out)

print("----- (3) obtain the 2nd to 7th elements with a step of 2 using “:” sign-----")
out = arr[1:6:2]
print("out:",out)






# %%
