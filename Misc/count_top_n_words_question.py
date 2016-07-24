"""Count words."""
def count_words(s, n):
    """Return the n most frequently occuring words in s."""

    # Count the number of occurences of each word in s
    words = s.split() # split words into list on spaces
    uniquewords = []

    # for every word in the list
    for word in words:
        iswordunique = True

        # check to see if that word has been seen before
        for ii, pair in enumerate(uniquewords):

            # if that word has been seen before
            if pair[0] == word:

                # update the count of that word
                uniquewords[ii] = (pair[0], pair[1] + 1)
                iswordunique = False
                break

        # if word wasn't seen, add it to the list
        if iswordunique:
            uniquewords.append((word,1)) 

    # Sort the occurences in descending order (alphabetically in case of ties)
    uniquewordsalph = sorted(uniquewords, key=lambda word: word[0])
    uniquewordsdesc = sorted(uniquewordsalph, key=lambda word: word[1], reverse=True)

    # Return the top n words as a list of tuples (<word>, <count>)
    return uniquewordsdesc[0:n]

def test_run():
    """Test count_words() with some inputs."""
    print count_words("cat bat mat cat bat cat", 3)
    print count_words("betty bought a bit of butter but the butter was bitter", 3)

if __name__ == '__main__':
    test_run()
