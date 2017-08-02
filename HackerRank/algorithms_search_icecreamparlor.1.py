# https://www.hackerrank.com/challenges/icecream-parlor

from bisect import bisect_left

def read_input():
    numTrips = int(input().strip())
    trips = []
    for ii in range(0, 2 * numTrips, 3):
        trip = {}
        trip["m"] = int(input().strip()) # money available
        trip["n"] = int(input().strip()) # number of flavors available
        trip["c"] = [int(i) for i in input().strip().split()] # cost of flavors
        trip["c_sorted"] = sorted(trip["c"]) # sorted cost of flavors
        trip["c_indexed"] = sorted(range(trip["c"], key = lambda k: trip["c"][k])) # just sort the indexes for mapping back later
        trips.append(trip)
    
    return numTrips, trips

def binary_search(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x: return i
    else: return None

def find_two_flavors(flavors, flavorsSorted, flavorsIndexed, money, numFlavors):
    for ii in range(numFlavors):

        # quickly search if this value is available using the sorted list
        available = binary_search(flavorsSorted, money - flavors[ii])

        # the matching flavor cost wasn't available
        if ind == None: pass

        # now that we know it's available, find the original index
        # using a linear search, but excluding the first N
        else:
            mapped = binary_search(flavorsIndexed, available)

            if mapped == ii:
                pass
            else:
                return mapped

    return [] # should never happen

# =============
# Program Start
# =============

numTrips, trips = read_input()

# pick flavors for each trip
flavors = []
for trip in trips:
    flavors.append(find_two_flavors(trip['c'], trip['c_sorted'], trip['c_indexed'], trip['m'], trip['n']))

# output answer
for trip in flavors:
    print("{}".format(' '.join(str(iD) for iD in trip)))