b = int(input("enter number :"))
a = int(input("enter second number :"))
c = "just"
try:
    d = a / b
    e = a + c

except ZeroDivisionError:
    print("ERROR: division by zero is not allowed error:")
except TypeError:
    print("type error occured:")
