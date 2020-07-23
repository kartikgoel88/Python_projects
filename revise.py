# python data structures
# lists tuple dict set strings nested lists
list_ex=[3,4,1,2,3,3]
dict_ex={'k' : 'v' }
tuple_ex=(1,2,3,3)
set_ex={1,2,3,3}
string_ex="1233"
print("objs: {} {} {} {}".format(list_ex,dict_ex,tuple_ex,set_ex))
# indexing
print("indexing: {} {} {}".format(list_ex[1],tuple_ex[1],string_ex[1]))
#  slicing
print("slicing: {} {} {}".format(list_ex[1:],tuple_ex[1:],string_ex[1:]))
# add item
dict_ex['a'] = 'b'
print("adding: {} {}".format(list_ex.append(10),dict_ex))
# delete item
print("deletion: {} {}".format(list_ex.pop(0),dict_ex.pop('k')))
# merge objs
# sorting
list_ex.sort()
print("sortin: {} ".format(list_ex))
print("objs: {} {} {} {}".format(list_ex,dict_ex,tuple_ex,set_ex))

# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(list_ex):
    # write your code in Python 3.6
    a_set=set(list_ex)
    min_set=min(a_set)
    max_set=max(a_set)
    b_set=set(range(min_set,max_set))
    print(b_set)
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


print("input:{}".format(list_ex))
print(solution([1,2,3]))
# pandas data structures
# series dataframes
#