import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # get the path of the folder
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)

    # get the names of all files or folders inside the given path
    file_names = os.listdir(path)
    files_dict = dict()

    # map the file names to contents
    for txt_file in file_names:
        f = open(os.path.join(path, txt_file), "r", encoding="utf8")
        files_dict[txt_file] = f.read()
        f.close()

    return files_dict

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    #tokenize the document and create a variable to store stop words
    tokens = nltk.word_tokenize(document)
    tokens_clean = list()
    stop_words = nltk.corpus.stopwords.words("english")
    punctuation = [p for p in string.punctuation]

    #iterate through the tokens and filter out the words that are punctuation or stop words
    for w in tokens:
        low = w.lower()
        if low not in punctuation and low not in stop_words:
            tokens_clean.append(low)


    return tokens_clean


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    idf_values = dict()
    counter = dict()
    num_docs = len(documents)

    for document in documents:
        for w in set(documents[document]):
            if w in counter.keys():
                counter[w] += 1
            else:
                counter[w] = 1


    for word, value in counter.items():
        idf_values[word] = math.log((num_docs / value))
    return idf_values


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    frequency = dict()
    freq_in_doc = dict()
    tf_idf = 0
    tf_idf_files = dict()
    for file in files:
        for query_word in query:
            if query_word not in files[file]:
                continue
            for word in files[file]:
                if query_word == word:
                    if word in frequency:
                        frequency[query_word] += 1
                    else:
                        frequency[query_word] = 1
        freq_in_doc[file] = frequency
        frequency = dict()
        #calculate tf-idf values of each term from the query present in file words
        for word, value in freq_in_doc[file].items():
            #tf-idf from the current file of the loop
            tf_idf = tf_idf + value * idfs[word]
            # mapping file to tf-idf value
            tf_idf_files[file] = tf_idf
        tf_idf = 0
    # sort the files by value
    tf_idf_files = sorted(tf_idf_files, key=tf_idf_files.get, reverse=True)
    return tf_idf_files[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_idf = dict()
    idf_sum = 0
    density = 0
    for sentence in sentences:
        for word in query:
            if word in sentences[sentence]:
                idf_sum = idf_sum + idfs[word]
                density += 1
            else:
                continue
        if density != 0:
            length = len(sentences[sentence])
            density = density / length
            sentence_idf[sentence] = (idf_sum, density)
            density = 0
            idf_sum = 0
    sentence_idf = sorted(sentence_idf, key=lambda k: (sentence_idf[k][0], sentence_idf[k][1]), reverse=True)
    return sentence_idf[:n]


if __name__ == "__main__":
    main()
