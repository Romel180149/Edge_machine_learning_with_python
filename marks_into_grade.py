def marks_into_grade(marks):
    if marks >= 80 and marks <= 100:
        return "A+"
    elif marks >= 75 and marks <= 79:
        return "A"
    elif marks >= 70 and marks <= 74:
        return "A-"
    elif marks >= 65 and marks <= 69:
        return "B+"
    elif marks >= 60 and marks <= 64:
        return "B"
    elif marks >= 55 and marks <= 59:
        return "B-"
    elif marks >= 45 and marks <= 49:
        return "C"
    else:
        return "F"


marks = int(input("Enter your marks:"))
grade = marks_into_grade(marks)
print(f"Your grade is {grade}")
