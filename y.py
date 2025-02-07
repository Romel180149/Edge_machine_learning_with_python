# fpr = open("E:\Edge_ML_With_python\b.csv", "r")
fpr = open(r"E:\Edge_ML_With_python\b.csv", "r")
txt =fpr.readline()
print(txt)
if (len(txt)<1):
    print("File end")
    fpr.close()