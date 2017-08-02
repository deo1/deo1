# https://www.hackerrank.com/challenges/icecream-parlor

def read_input():
    numTrips = int(input().strip())
    trips = []
    for ii in range(0, 2 * numTrips, 3):
        trip = {}
        trip["m"] = int(input().strip()) # money available
        trip["n"] = int(input().strip()) # number of flavors available
        trip["c"] = [int(i) for i in input().strip().split()] # cost of flavors
        trips.append(trip)
    
    return numTrips, trips

# slow
def find_two_flavors_slow(flavors, money, numFlavors):
    for ii in range(numFlavors):
        if flavors[ii] < money:
            for jj in range(ii + 1, numFlavors):
                if flavors[jj] == money - flavors[ii]:
                    return [ii + 1, jj + 1]

    return [] # should never happen

def generate_test_cases(N, M, P):
    from random import randint

    cases = []
    for ii in range(N):
        case = {}
        case['c'] = [randint(1, M) for cost in range(P)]
        case['n'] = len(case['c'])
        flavor1 = randint(1, P - 1)
        flavor2 = randint(1, P - 1)
        while flavor1 == flavor2: flavor2 = randint(1, P - 1)
        case['m'] = case['c'][flavor1] + case['c'][flavor2]
        cases.append(case)
    
    return cases

test = False

# =============
# Program Start
# =============

if test:
    trips = generate_test_cases(1000, 1000, 900)
else:
    numTrips, trips = read_input()

# pick flavors for each trip
flavors = []
for trip in trips:
    flavors.append(find_two_flavors_slow(trip['c'], trip['m'], trip['n']))

# output answer
ii = 0
for flav in flavors:
    print("{}".format(' '.join(str(iD) for iD in flav)))

    if test:
        valA = trips[ii]['c'][flav[0] - 1]
        valB = trips[ii]['c'][flav[1] - 1]
        if valA + valB == trips[ii]['m']:
            print("Pass\n")
        else:
            print("Fail\n")
            raise RuntimeError
    
    ii = ii + 1