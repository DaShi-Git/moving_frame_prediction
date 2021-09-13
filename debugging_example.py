'''
Tutorial 1 (12 April 2021)
Practical Course: Learning For Self-Driving Cars and Intelligent Systems
Summer Semester 2021
Technical University of Munich
https://vision.in.tum.de/teaching/ss2021/intellisys_ss2021
'''

import pdb
sum = 0
for x in range(10):
    print("Value of variable x:", x)
    if x == 4:
        pdb.set_trace()
    sum += x
print("End of loop")
print("End of script")


#n → next line
#c → next occurrence of pdb
#l → display +/- 5 lines around the currently to be executed command
#q → Quit script
#Don’t name your variables with these shortcut names while debugging.

