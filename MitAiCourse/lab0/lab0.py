# This is the file you'll use to submit most of Lab 0.

# Certain problems may ask you to modify other files to accomplish a certain
# task. There are also various other files that make the problem set work, and
# generally you will _not_ be expected to modify or even understand this code.
# Don't get bogged down with unnecessary work.

import deo1tests

# Section 2: Programming warmup _____________________________________________

# Problem 2.1: Warm-Up Stretch
def cube(x):
    return x ** 3

def factorial(x):
    if x < 0:
        raise RuntimeError("factorial() : input must not be negative")
    elif not isinstance(x, int):
        raise TypeError("factorial() : input must be an integer")

    # special case per mathematical definition of factorial
    elif x == 0:
        return 1

    # recursive base case
    elif x == 1:
        return 1

    # expand the product
    else:
        total = x * factorial(x - 1)
        return total

def count_pattern(pattern, lst):
    """
    count_pattern() returns the count of the number of times that the pattern occurs in the lst
    """
    count = 0
    if type(pattern) != type(lst):
        raise TypeError("count_pattern() : arguments must be of the same type")
    elif not pattern or not lst:
        return count
    else:

        # size of the pattern and the lst
        patternlength = len(pattern)
        lstlength = len(lst)

        # if the pattern is longer than the lst, quit out
        if patternlength > lstlength:
            return count

        # otherwise look for matches
        else:

            # for the maximum total possible matches
            for ii in range(lstlength - patternlength + 1):

                # step the pattern through the lst
                candidatematch = lst[ii:(ii + patternlength)]

                # if it's a match, increment the count of the matches
                if pattern == candidatematch:
                    count += 1
            return count


# Problem 2.2: Expression depth
def depth(expr):
    """
    depth(expr) finds the depth of the mathematical expr formatted as a list.
    depth is defined as the deepest level of nested operations within the expression.
    """
    if not isinstance(expr, list):
        raise TypeError("depth() : expr must be of type list")
    else:
        exprlength = len(expr) # number of nodes at current recursion level
        depths = [0]*exprlength # depth beneath each node at the current level

        # for every node at the current level
        for ii in range(exprlength):

            # if the node has branches
            if isinstance(expr[ii], list):

                # increment depth and recurse down the branch
                depths[ii] = 1 + depth(expr[ii])

        # pass up the deepest depth beneath the current level
        return max(depths)


# Problem 2.3: Tree indexing
def tree_ref(tree, index):
    """
    tree_ref(tree, index) returns the tree or leaf at the index, where index is
    in the form of (branch1int, branch2int, ..., branchNint)
    """

    if not isinstance(tree, list):
        raise TypeError("tree_ref : tree must be of type list")
    elif not isinstance(index, list):
        raise TypeError("tree_ref : index must be of type list")
    elif depth(index) != 0:
        raise TypeError("tree_ref : index must be a flat list")
    else:

        # recursive base case, reached the final node
        if len(index) == 1:
            return tree[index[0]]

        # recurse into tree at current index, truncating index++
        else:
            subtree = tree_ref(tree[index[0]], index[1:])

    # not recursive base case, pass up nested return value
    return subtree


# Section 3: Symbolic algebra

# Your solution to this problem doesn't go in this file.
# Instead, you need to modify 'algebra.py' to complete the distributer.

# from algebra import Sum, Product, simplify_if_possible
# from algebra_utils import distribution, encode_sumprod, decode_sumprod


# Run Tests
if __name__ == '__main__':
    deo1tests.test_lab0()
