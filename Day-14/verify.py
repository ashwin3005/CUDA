import numpy as np
import torch

def string_to_array(mat_str):
    mat = [
        list(map(float, line.split(" ")))
        for line in mat_str.strip().split("\n")
    ]
    return np.array(mat)

a_string = """1.000000 2.000000 3.000000 
4.000000 5.000000 6.000000 
7.000000 8.000000 9.000000 
10.000000 11.000000 12.000000"""

b_string = """1.000000 3.000000 5.000000 7.000000 9.000000 11.000000 
13.000000 15.000000 17.000000 19.000000 21.000000 23.000000 
25.000000 27.000000 29.000000 31.000000 33.000000 35.000000"""

a_array = string_to_array(a_string)

b_array = string_to_array(b_string)


a = np.array(a_array)


b = np.array(b_array)

a @ b