# import numpy as np

# A = np.array([5, 3, 8, 9, 6])
# A = -np.sort(-A)
# print(type(A))
# print(A)
# read ,write, append
# file pointer raw,package ;xl,csv
# fpr = open(r"E:\b.csv","r")
# import os

# file_path = r"E:\b.csv"
# if os.path.exists(file_path):
#     fpr = open(file_path, "r")
#     # Do something with the file
# else:
#     print(f"The file {file_path} does not exist.")
fpr = open(r"E:\Edge_ML_With_python\b.csv", "r")
# import csv

# file_path = r"E:\b.csv"

# try:
#     with open(file_path, "r") as fpr:
#         csv_reader = csv.reader(fpr)
        
#         # Read and print header
#         header = next(csv_reader, None)
#         print("Header:", header)  # This will print the header row
        
#         # Read and print the rest of the rows
#         for row in csv_reader:
#             print(row)  # Print each row
# except FileNotFoundError:
#     print(f"The file {file_path} does not exist.")
# except Exception as e:
#     print(f"An error occurred: {e}")
# file_path =open( r"E:\b.csv","r")



