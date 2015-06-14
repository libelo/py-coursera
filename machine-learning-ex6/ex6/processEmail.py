from os import path
import re
from porter import PorterStemmer
from numpy import array

def getVocabDict():
    #GETVOCABDICT reads the fixed vocabulary list in vocab.txt and returns a
    #dictionary mapping words to integers
    #   vocabList = GETVOCABDICT() reads the fixed vocabulary list in vocab.txt
    #   and returns a dictionary of the words.

    ## Read the fixed vocabulary list
    with open('vocab.txt') as f:
        # Store all dictionary words in a python dict which maps strings to integers
        vocab = {}
        for line in f:
            id, word = line.split()
            vocab[word] = int(id)

    return vocab

def processEmail(email_contents):
    #PROCESSEMAIL preprocesses a the body of an email and
    #returns a list of word_indices
    #   word_indices = PROCESSEMAIL(email_contents) preprocesses
    #   the body of an email and returns a list of indices of the
    #   words contained in the email.
    #

    # Load Vocabulary
    vocab = getVocabDict()

    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================

    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers

    # hdrstart = email_contents.find('\n\n')
    # email_contents = email_contents[hdrstart+2:]

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with >
    # and does not have any < or > in the tag and replace it with a space
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)


    # ========================== Tokenize Email ===========================

    # Output the email to screen as well
    print '\n==== Processed Email ====\n'

    # Process file
    l = 0
    porterStemmer = PorterStemmer()
    # Tokenize and also get rid of any punctuation
    sep = '[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\},\'\"\>\_\<\;\%\n\r]+'
    for s in re.split(sep, email_contents):
        # Remove any non alphanumeric characters
        s = re.sub('[^a-zA-Z0-9]', '', s)

        # Stem the word
        s = porterStemmer.stem(s.strip())

        # Skip the word if it is too short
        if len(s) < 1:
           continue

        # Look up the word in the dictionary and add to word_indices if
        # found
        # ====================== YOUR CODE HERE ======================
        # Instructions: Fill in this function to add the index of str to
        #               word_indices if it is in the vocabulary. At this point
        #               of the code, you have a stemmed word from the email in
        #               the variable s. You should look up s in the
        #               vocabulary dictionary (vocab). If a match exists, you
        #               should add the index of the word to the word_indices
        #               vector. Concretely, if s = 'action', then you should
        #               add to word_indices the value under the key 'action'
        #               in vocab. For example, if vocab['action'] = 18, then,
        #               you should add 18 to the word_indices vector
        #               (e.g., word_indices.append(18) ).
        #




        # =============================================================


        # Print to screen, ensuring that the output lines are not too long
        if l + len(s) + 1 > 78:
            print
            l = 0
        print s,
        l += len(s) + 1

    # Print footer
    print '\n========================='

    return array(word_indices)
