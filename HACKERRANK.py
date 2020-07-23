import math

x=int(input())
numbers = list(map(int,raw_input().split(" ")))
print(numbers)

def twoStrings_logic(s1, s2):
    flag = 'NO'
    for i in s1:
        if i in list(s2):
            flag = 'YES'
    return flag

def twoStrings_set(s1, s2):
    flag = 'NO'
    ss1 = set(s1)
    ss2 = set(s2)

    if bool(ss1.intersection(ss2)):
        flag ='YES'
    return flag

def alternatingCharacters(s):
    counter =0
    for i in range(0,len(s)):
        try:
            print(s[i],s[i+1],counter)
            if s[i] == s[i+1]:
                counter += 1
            continue
        except:
            continue
    return counter

class Player:
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def __repr__(self):
        pass

    def comparator(a, b):

        if a.score == b.score:
            if a.name < b.name:
                return -1
            else:
                return 1
        elif a.score < b.score:
            return 1
        else:
            return -1

# Complete the pairs function below.
def pairs(k, arr):
    arr_1 = []
    for i in arr:
        arr_1.append(i+k)

    return len(list(set(arr).intersection(arr_1)))

def gradingStudents(grades):
    # Write your code here
    rou_gr =[]
    for i in grades:
        if i < 38:
            rou_gr.append(i)
        elif (5 - (i % 5)) >= 3:
            rou_gr.append(i)
        elif (5 - (i % 5)) < 3:
            rou_gr.append(i+(5 - (i % 5)))
    return rou_gr

def findGeometricTriplets_logic(arr, r):
    trips =[]
    n = len(arr)
    for i in range(0, n -2):
        j=i+1
        k=j+1
        while j < n-1 :
            while k < n :
                print(i,j,k)
                if (arr[j] % arr[i] == 0 and
                    arr[k] % arr[j] == 0 and
                    arr[j] // arr[i] == r and
                    arr[k] // arr[j] == r):

                    trips.extend([ [arr[i] , arr[j] , arr[k]]])
                k += 1
            j += 1
            k = j +1
    print(trips)

def findGeometricTriplets(arr, r):
    trips =[]
    n = len(arr)
    m2 = {}
    m3 = {}
    cnt = {}
    for i in arr:

        if i*r in arr:
            m2[i] = i*r
            #cnt[i*r] = arr.count(i*r)
        if i*r*r in arr:
            m3[i] = i*r*r
            #cnt[i*r*r] = arr.count(i*r*r)
        cnt[i] = arr.count(i)
    print(m2,m3,cnt)

    for i in reversed(arr):
        if i in m2 and i in m3:
            #if cnt[i] == 1 and cnt[m2[i]] ==1 and cnt[m3[i]] ==1 :
            trips.extend([[i,m2[i],m3[i]]])
    print(trips)


# findGeometricTriplets([1,3,9,9,27,27,81],3)



