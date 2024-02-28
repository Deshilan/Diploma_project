import os
import shutil

dir_S = r'C:\python\PROJEKT_DYPLOMOWY\DATASETS_BEFORE_RANDOMIZATION\ACTUAL_DATASET_128\0'
dir_D = r'C:\python\PROJEKT_DYPLOMOWY\ACTUAL_DATASET_128_CLEANED\0'
CLEANED = r"C:\python\PROJEKT_DYPLOMOWY\ACTUAL_DATASET_64_CLEANED\0"

os.chdir(dir_S)
for file in os.listdir(dir_S):
    move = 0
    for filename in os.listdir(CLEANED):
        if filename == file:
            move = 1
    if move == 1:
        print(file)
        shutil.copyfile(file, dir_D+file)


dir_S = r'C:\python\PROJEKT_DYPLOMOWY\DATASETS_BEFORE_RANDOMIZATION\ACTUAL_DATASET_192\0'
dir_D = r'C:\python\PROJEKT_DYPLOMOWY\ACTUAL_DATASET_192_CLEANED\0'
CLEANED = r"C:\python\PROJEKT_DYPLOMOWY\ACTUAL_DATASET_64_CLEANED\0"

os.chdir(dir_S)
for file in os.listdir(dir_S):
    move = 0
    for filename in os.listdir(CLEANED):
        if filename == file:
            move = 1
    if move == 1:
        print(file)
        shutil.copyfile(file, dir_D+file)
