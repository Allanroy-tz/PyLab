# %%
from array import array
from sqlite3 import Row
from tkinter.tix import COLUMN
import numpy as np

# check the number of dimensions,shape,size and data type of data
# 测试 pull
data = np.arange(12).reshape(4, 3)
print(data)
print("-----Create a 3-by-4 zero array-----")
data01 = np.zeros(12).reshape(3, 4)

print(data01)

print("-----Create a 3-by-4 one array-----")
data02 = np.ones(12).reshape(3, 4)
print(data02)

print("-----Create a 5-by-2 empty array-----")
data03 = np.empty(10).reshape(5, 2)
print(data03)

print("-----Create an array array([ 1, 6, 11, 16]) using arrange-----")
data04 = np.arange(1, 20, 5)
print(data04)

print("-----create a 2-by-3 zero array by specifying the data type as float64-----")
data05 = np.zeros(6, np.float64)
print(data05)
print("DTpey=", data05.dtype.name)

print("-----Show the name of the data type of the array-----")
data_one = np.array([[1, 2, 3], [4, 5, 6]])
print(data_one)
print("data_one.DTpey=", data_one.dtype.name)

print("-----Change the data type of data_one to float64-----")
data_one = data_one.astype(np.float64)
print(data_one)
print("data_one.DTpey=", data_one.dtype.name)

print(
    "-----Change the data type of float_data = np.array([1.2, 2.3, 3.5]) to int64-----")
float_data = np.array([1.2, 2.3, 3.5]).astype(np.int64)
print(float_data)
print("float_data.DTpey=", float_data.dtype.name)

print(
    "-----Change the data type of str_data = np.array(['1', '2', '3']) to int64-----")
str_data = np.array(['1', '2', '3']).astype(np.int64)
print(str_data)
print("str_data.DTpey=", str_data.dtype.name)

print("-----Create two arrary:data1,data2-----")
data1 = np.array([[1, 2, 3], [4, 5, 6]])
data2 = np.array([[1, 2, 3], [4, 5, 6]])
print("data1:\n", data1)
print("data2:\n", data2)

print("-----(1) data1 + data2-----")
data_out = data1+data2
print("data_out:\n", data_out)

print("-----(2) data1 * data2-----")
data_out = data1*data2
print("data_out:\n", data_out)

print("-----(3) data1 - data2-----")
data_out = data1-data2
print("data_out:\n", data_out)

print("-----(4) data1 / data2-----")
data_out = data1/data2
print("data_out:\n", data_out)

print("-----Create two arrary:arr1,arr2-----")
arr1 = np.array([[0], [1], [2], [3]])
arr2 = np.array([1, 2, 3])
print("arr1:\n", arr1)
print("arr2:\n", arr2)

print("-----Calculate arr1+arr2-----")
arr_out = arr1+arr2
print("arr_out:\n", arr_out)

print("-----Create two arrary:data1,data2-----")
data1 = np.array([[1, 2, 3], [4, 5, 6]])
data2 = 10
print("data1:\n", data1)
print("data2:\n", data2)


print("-----(1) data1 + data2-----")
data_out = data1+data2
print("data_out:\n", data_out)

print("-----(2) data1 * data2-----")
data_out = data1*data2
print("data_out:\n", data_out)

print("-----(3) data1 - data2-----")
data_out = data1-data2
print("data_out:\n", data_out)

print("-----(4) data1 / data2-----")
data_out = data1/data2
print("data_out:\n", data_out)

print("-----Create an array as arr = np.arange(8)-----")
arr = np.arange(0, 8, 1)
print("arr:\n", arr)

print("----- (1) obtain the 6th element-----")
out = arr[5]
print("out:", out)

print("----- (2) obtain the 4th to 5th elements using “:” sign-----")
out = arr[3:4:1]
print("out:", out)

print("----- (3) obtain the 2nd to 7th elements with a step of 2 using “:” sign-----")
out = arr[1:6:2]
print("out:", out)


print(
    "-----Create an array as arr2d = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])-----")
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("arr2d:\n", arr2d)

print("----- (1) obtain the 2nd row-----")
out = arr2d[1, ...]
print("out:", out)

#TODO: 啥意思
print("----- (2) obtain the element of the 1st row and the 2nd column-----")
# out = arr2d[]
print("out:", out)

print("----- (3) obtain the first two rows using “:” sign-----")
out = arr2d[:2]
print("out:", out)

print(
    "----- (4) obtain the array as array([[1, 2], [4, 5]]), using “:” sign-----")
out = arr2d[0:2, 0:2]
print("out:", out)

print("----- (5) obtain the array as array([4, 5]), using “:” sign-----")
out = arr2d[1:2, 0:2]
print("out:", out)

print("-----Create an array of //Obtain the two arrays using fancy indexing-----")
arr4d = np.array([[0., 1., 2., 3.], [1., 2., 3., 4.],
                 [2., 3., 4., 5.], [3., 4., 5., 6.]])
print("arr4d:\n", arr4d)

print("----- (1) array([[0., 1., 2., 3.],[2., 3., 4., 5.]])-----")
out = arr4d[[0, 2]]
print("out:\n", out)

print("----- (2) array([2., 5.])-----")
out = arr4d[[0, 2], [2, 3]]
print("out:\n", out)

print("-----Create an array of names-----")
student_name = np.array(["Tom", "Lily", "Jack", "Rose"])
print("student_name:\n", student_name)

print("-----Create a score matrix-----")
student_score = np.array(
    [[79, 88, 80], [89, 90, 92], [83, 78, 85], [78, 76, 80]])
print("student_name:\n", student_score)

print("----- Obtain the Jack’s score using a bool array-----")
out = student_score[[False, False, True, False]]
print("out:\n", out)

print("-----Create an array-----")
arr = np.arange(16).reshape(2, 2, 4)
print("arr:\n", arr)

print("----- (1) Perform the transpose using two methods mentioned in the class-----")
out = arr.T
print("out:\n", out)
out = arr.transpose()
print("out:\n", out)

print("----- (2) Try arr.transpose(1, 2, 0) and write down your understanding of this operation-----")
out = arr.transpose(1, 2, 0)
print("out:\n", out)
print("自己的理解：\n0,1,2 依次代表三个方向的维度,transpose(0,1,2)这是三个维度与三个参数绑定的初始的位置，如果变换了三个参数的位置，就代表相应的维度与原来的交换了\n")

#TODO: 翻转数组还是有点模糊
print("----- (3) Swap the axis 1 and 0-----")
out = arr.swapaxes(1,0)
print("out:\n", out)

print("-----Create an array -----")
x = np.array([12, 9, 13, 15])
print("x:", x)

print("-----(1) calculate the square root of x-----")
out = np.sqrt(x)
print("out:", out)

print("-----(2) calculate the absolute value of x-----")
out = np.abs(x)
print("out:", out)

print("-----(3) calculate the square of x-----")
out = np.square(x)
print("out:", out)

print("-----Create two arrays: arr_x,arr_y -----")
arr_x = np.array([1,5,7])
arr_y = np.array([2,6,8])
print("arr_x:", arr_x)
print("arr_y:", arr_y)

print("-----Use np.where() to obtain the array array([1, 6, 7])-----")
arr_con = np.array([True,False,True])
out = np.where(arr_con,arr_x,arr_y)
print("out:", out)

print("-----Create an array -----")
arr = np.arange(10)
print("arr:", arr)

print("-----(1) Calculate the summation of the elements in arr-----")
out = np.sum(arr)
print("out:", out)

print("-----(2) Calculate the average-----")
out = np.average(arr)
print("out:", out)

print("-----(3) Calculate the minimal-----")
out = np.amin(arr)
print("out:", out)

print("-----(4) Calculate the maximal-----")
out = np.amax(arr)
print("out:", out)


print("-----Create an array -----")
arr = np.array([[6,2,7],[3,6,2],[4,3,2]])
print("arr:", arr)

print("-----(1) Sort each row-----")
out = np.sort(arr,1)
print("out:", out)

print("-----(2) Sort each column-----")
out = np.sort(arr,0)
print("out:", out)

print("-----Create an array -----")
arr = np.array([[1,-2,-7],[-3,6,2],[-4,3,2]])
print("arr:", arr)

print("-----(1) Check if all the elements is greater than 0-----")
out = np.all(arr>0)
print("if all the elements is greater than 0? :", out)

print("-----(2) Check if at least one element is greater than 0-----")
out = np.any(arr>0)
print("if at least one element is greater than 0?:", out)

print("-----Create an array -----")
arr = np.array([12, 11, 34, 23, 12, 8, 11])
print("arr:", arr)

print("-----(1) find unique values-----")
out = np.unique(arr)
print("out:", out)

print("-----(2) each element of arr is also present in the array [11,12]-----")
out = np.in1d(arr,[11,12])
print("out:", out)

print("-----Create two matrix : arr_x,arr_y -----")
arr_x = np.array([[1, 2, 3], [4, 5, 6]])
arr_y = np.array([[1, 2], [3, 4], [5, 6]])
print("arr_x:\n", arr_x)
print("arr_y:\n", arr_y)

print("-----Calculate the matrix multiplication using three methods-----")
print("----- (1) Using dot -----")
out = np.dot(arr_x,arr_y)
print("out:\n", out)

print("----- (2) Using vdot -----")
out = np.vdot(arr_x,arr_y)
print("out:", out)

print("----- (3) Using matmul -----")
out = np.matmul(arr_x,arr_y)
print("out:", out)