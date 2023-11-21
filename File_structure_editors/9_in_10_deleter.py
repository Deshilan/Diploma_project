import os
import shutil


dest0 = r'C:\python\pythonProject\My_dataset\0'
dest1 = r'C:\python\pythonProject\My_dataset\1'
source0 = r'C:\python\pythonProject\Extra_pictures_to_dataset\0'
source1 = r'C:\python\pythonProject\Extra_pictures_to_dataset\1'

os.chdir(source0)
for file in os.listdir(source0):
    shutil.move(file, dest0)

os.chdir(source1)
for file in os.listdir(source1):
    shutil.move(file, dest1)