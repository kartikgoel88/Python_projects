import matplotlib
import pandas_datareader.data

R4udWToTe_86q8d595We

def solution(A):
    # write your code in Python 3.6
    print(A)
    a_set=set(A)
    print("a_set:".format(a_set))
    min_set=min(a_set)
    print("min_set:".format(min_set))
    max_set=max(a_set)
    b_set=set(range(min_set,max_set))
    print("b_set:".format(b_set))
    c_set=b_set - a_set
    print(c_set)
    if c_set is None:
        val= max(c_set) + 1
        if val == 0:
            return 1
        else:
            return val

    else:
        return max(c_set)
if __name__ == "__main__" :
    print("starting")
    solution([2])