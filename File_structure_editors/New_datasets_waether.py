import os
import shutil

directory = 'C:\python\pythonProject\My_dataset/0'
directory2 = 'C:\python\pythonProject\My_dataset/1'
destination1 = 'C:\python\pythonProject\Test_datasets\Sunny0'
destination2 = 'C:\python\pythonProject\Test_datasets\Rainy/0'
destination3 = 'C:\python\pythonProject\Test_datasets\Sunny/1'
destination4 = 'C:\python\pythonProject\Test_datasets\Rainy/1'

for filename in os.listdir(directory):
    if filename[0]== "S":
        shutil.copy( 'C:\python\pythonProject\My_dataset/0/'+filename, destination1)
    if filename[0] == "R":
        shutil.copy('C:\python\pythonProject\My_dataset/0/'+filename, destination2)

for filename in os.listdir(directory2):
    if filename[0]== "S":
        shutil.copy('C:\python\pythonProject\My_dataset/1/'+filename, destination3)
    if filename[0] == "R":
        shutil.copy('C:\python\pythonProject\My_dataset/1/'+filename, destination4)