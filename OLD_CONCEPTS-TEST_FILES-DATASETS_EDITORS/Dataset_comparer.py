import os

x = 0
os.chdir(r'C:\python\PROJEKT_DYPLOMOWY\TEST_DATASET\256x256\0')
for file in os.listdir(r'C:\python\PROJEKT_DYPLOMOWY\TEST_DATASET\256x256\0'):
    i = 1
    for filename in os.listdir(r'C:\python\PROJEKT_DYPLOMOWY\TEST_DATASET\64x64\0'):
        if filename == file:
            i = 0
            break
    if i == 1:
        os.remove(file)
        x = x + 1

print(x)