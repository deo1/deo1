import lab0
cube = lab0.cube
factorial = lab0.factorial
count_pattern = lab0.count_pattern
depth = lab0.depth
tree_ref = lab0.tree_ref

def test_lab0():
    """
    # test cube()
    """
    print("\n----- Testing cube() -----")
    x = 3
    print(cube(x))

    """
    # test factorial()
    """
    print("\n----- Testing factorial() -----")
    x = -1
    try:
        print(factorial(x))
    except RuntimeError as e:
        print("Function returned an error for input " + str(x) + " : " + str(e))

    x = 22.34234
    try:
        print(factorial(x))
    except TypeError as e:
        print("Function returned an error for input " + str(x) + " : " + str(e))

    x = 0
    print(factorial(x))

    x = 1
    print(factorial(x))

    x = 3
    print(factorial(x))

    x = 6
    print(factorial(x))

    """
    # test count_pattern()
    """
    print("\n----- Testing count_pattern() -----")
    pattern = ('a', 'b')
    lst = ['a', 'b', 'c', 'f', 'a', 'b', 'a']
    CA = None
    try:
        N = count_pattern(pattern, lst)
        print("Correct answer is " + str(CA) + " returned answer is " + str(N))
    except TypeError as e:
        print("Function correctly returned an error : " + str(e))

    pattern = {'a', 'b'}
    lst = ['a', 'b', 'c', 'f', 'a', 'b', 'a']
    CA = None
    try:
        N = count_pattern(pattern, lst)
        print("Correct answer is " + str(CA) + " returned answer is " + str(N))
    except TypeError as e:
        print("Function correctly returned an error : " + str(e))

    pattern = {'a', 'b'}
    lst = {'a', 'b', 'c', 'f', 'a', 'b', 'a'}
    CA = None
    try:
        N = count_pattern(pattern, lst)
        print("Correct answer is " + str(CA) + " returned answer is " + str(N))
    except TypeError as e:
        print("Function correctly returned an error : " + str(e))

    pattern = ['a', 'b']
    lst = ['a', 'b', 3, 'f', 'a', 'b', 'a']
    CA = 2
    N = count_pattern(pattern, lst)
    print("Correct answer is " + str(CA) + " : returned answer is " + str(N))

    pattern = ['a', 'b']
    lst = ['a', 'b', 'c', 'f', 'a', 'b', 'a']
    CA = 2
    N = count_pattern(pattern, lst)
    print("Correct answer is " + str(CA) + " : returned answer is " + str(N))

    pattern = ['a', 'b', 'a']
    lst = ['a', 'b', 'c', 'f', 'a', 'b', 'a']
    CA = 1
    N = count_pattern(pattern, lst)
    print("Correct answer is " + str(CA) + " : returned answer is " + str(N))

    pattern = ['a', 'b', 'a']
    lst = ['g', 'a', 'b', 'a', 'b', 'a', 'b', 'a']
    CA = 3
    N = count_pattern(pattern, lst)
    print("Correct answer is " + str(CA) + " : returned answer is " + str(N))

    pattern = ['ab', 'a']
    lst = ['g', 'a', 'b', 'a', 'b', 'a', 'b', 'a']
    CA = 0
    N = count_pattern(pattern, lst)
    print("Correct answer is " + str(CA) + " : returned answer is " + str(N))

    """
    test depth(expr)
    """
    print("\n----- Testing depth(expr) -----")
    expr = ['x']
    CA = 0
    N = depth(expr)
    print("Correct answer is " + str(CA) + " : returned answer is " + str(N))

    expr = [['expt', 'x', '2']]
    CA = 1
    N = depth(expr)
    print("Correct answer is " + str(CA) + " : returned answer is " + str(N))

    expr = [['+', ['expt', 'x', 2], ['expt', 'y', 2]]]
    CA = 2
    N = depth(expr)
    print("Correct answer is " + str(CA) + " : returned answer is " + str(N))

    expr = [['/', ['expt', 'x', 5], ['expt', ['-', ['expt', 'x', 2], 1], ['/', 5, 2]]]]
    CA = 4
    N = depth(expr)
    print("Correct answer is " + str(CA) + " : returned answer is " + str(N))

    """
    test tree_ref()
    """
    print("\n----- Testing tree_ref() -----")
    tree = (1, (1, 2))
    index = [1]
    CA = None
    try:
        N = tree_ref(tree, index)
        print("Correct answer is " + str(CA) + " : returned answer is " + str(N))
    except TypeError as e:
        print("Function correctly returned an error : " + str(e))

    tree = [1, [1, 2]]
    index = (1)
    CA = None
    try:
        N = tree_ref(tree, index)
        print("Correct answer is " + str(CA) + " : returned answer is " + str(N))
    except TypeError as e:
        print("Function correctly returned an error : " + str(e))

    tree = [1, [1, 2]]
    index = [1, [2, 3]]
    CA = None
    try:
        N = tree_ref(tree, index)
        print("Correct answer is " + str(CA) + " : returned answer is " + str(N))
    except TypeError as e:
        print("Function correctly returned an error : " + str(e))

    tree = [1, [1, 2]]
    index = [0]
    CA = 1
    try:
        N = tree_ref(tree, index)
        print("Correct answer is " + str(CA) + " : returned answer is " + str(N))
    except TypeError as e:
        print("Function correctly returned an error : " + str(e))

    tree = [1, [1, 2]]
    index = [1,1]
    CA = 2
    try:
        N = tree_ref(tree, index)
        print("Correct answer is " + str(CA) + " : returned answer is " + str(N))
    except TypeError as e:
        print("Function correctly returned an error : " + str(e))

    tree = [[[1, 2], 3], [4, [5, 6]], 7, [8, 9, 10]]
    index = [3, 1]
    CA = 9
    try:
        N = tree_ref(tree, index)
        print("Correct answer is " + str(CA) + " : returned answer is " + str(N))
    except TypeError as e:
        print("Function correctly returned an error : " + str(e))

    tree = [[[1, 2], 3], [4, [5, 6]], 7, [8, 9, 10]]
    index = [1, 1, 1]
    CA = 6
    try:
        N = tree_ref(tree, index)
        print("Correct answer is " + str(CA) + " : returned answer is " + str(N))
    except TypeError as e:
        print("Function correctly returned an error : " + str(e))

    tree = [[[1, 2], 3], [4, [5, 6]], 7, [8, 9, 10]]
    index = [0,]
    CA = [[1, 2], 3]
    try:
        N = tree_ref(tree, index)
        print("Correct answer is " + str(CA) + " : returned answer is " + str(N))
    except TypeError as e:
        print("Function correctly returned an error : " + str(e))
