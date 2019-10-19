import numpy as np
from math import sqrt

def readDM(dm_file):
    """
    Read word vectors from a .dm file, where the first word of a line is the actual word
    and all other elements, separated by " " or "\t", are the word's vector.
    'DM' originally stands for 'dense matrix', but this requirement is not kept consistent.
    :param dm_file: str -- file path to a file with words and their vectors
    :return: {str:ndarray[float]} -- words and their vectors
    """
    dm_dict = {}
    with open(dm_file, "r") as f:
        dmlines=f.readlines()

    for l in dmlines:
        items=l.rstrip().split()
        row=items[0]
        vec=[float(i) for i in items[1:]]
        vec=np.array(vec)
        dm_dict[row]=vec
    return dm_dict

def readCols(cols_file):
    """
    Read a .cols file (normally correspondingly to a .dm file) and return two dictionaries:
    word:index and index:word. In the .cols file, the line number is used as index.
    :param cols_file: str -- file path to a list of words (words' line numbers are their index)
    :return: {int:str} -- 1:1 mapping of indices to words
    :return: {str:int} -- 1:1 mapping of words to indices
    """
    i_to_cols = {}
    cols_to_i = {}
    # line count
    c = 0
    with open(cols_file,'r') as f:
        for l in f:
            l = l.rstrip('\n')
            i_to_cols[c] = l
            cols_to_i[l] = c
            c+=1
    return i_to_cols, cols_to_i

def readDH(dh_file):
    """
    Read a file containing hash signatures in dense representation (file extension: .dh)
    .dh files, like .dm files, start with a word followed by numbers (in this case: all integers).
    In dense representation, the numbers of a hash signature represent the indices which are 1 in
    the binary (i.e., sparse) representation of that hash signature.
    :param dh_file: str -- file path to a file with words and their (dense) vectors
    :returns: {str:[int]} -- mapping of words to their dense hash signatures
    """
    dh_dict = readDM(dh_file)
    for w,h in dh_dict.items():
        dh_dict[w] = [int(v) for v in h]
    return dh_dict

def writeDH(sparse_space, dh_file):
    """
    Converts a hashed space (sparse representation) into dense representation
    and writes it to a file with the extension .dh
    :param sparse_space: {str:ndarray[int]} -- words and their binary hash signatures
    :param dh_file: str -- file path; should have the extension .dh
    """
    dense_space = {w:np.nonzero(h)[0] for w,h in sparse_space.items()}
    with open(dh_file, "w") as f:
        for w,h in dense_space.items():
            vectorstring = " ".join([str(v) for v in h])
            f.write("{0} {1}\n".format(w, vectorstring))

def sparsifyDH(dense_space, dims):
    """
    Produce the sparse representations (only 0s and 1s) of dense hashes.
    Each number in the dense hash is an index at which the sparse hash is 1.
    The size of the returned sparse hashes must be be provided.
    :param dense_space: {str:[int]} -- words and their dense hash signatures
    :param dims: int -- length of the sparse hash signatures to be produced
    :return: {str:ndarray[int]} -- words and their sparse (i.e., binary) hash signatures
    """
    sparse_space = {}
    for w,h in dense_space.items():
        sv = np.zeros(shape=(dims,))
        for i in h:
            sv[i] = 1
        sparse_space[w] = sv
    return sparse_space

def cosine_similarity(v1, v2):
    """
    :param v1: ndarray[float] -- vector 1
    :param v2: ndarray[float] -- vector 2
    :return: float -- cosine similarity of v1 and v2 (between 0 and 1)
    """
    if len(v1) != len(v2):
        return 0.0
    num = np.dot(v1, v2)
    den_a = np.dot(v1, v1)
    den_b = np.dot(v2, v2)
    return num / (sqrt(den_a) * sqrt(den_b))

def neighbours(space,word,n):
    """
    Find the n closest neighbours (by cosine) to a word in a space.
    :param space: {str:ndarray[float]} -- words and their vectors
    :param word: str
    :param n: int -- number of nearest neighbors to be returned
    :return: [str] -- words that are closest (by cosine distance)
    """
    cosines={}
    vec = space[word]
    for k,v in space.items():
        cos = cosine_similarity(vec, v)
        cosines[k]=cos

    neighbours = sorted(cosines, key=cosines.get, reverse=True)[:n]
    return neighbours

def simplify_postags(tagged_words):
    """
    Convert part-of-speech tags (Penn Treebank tagset) to the 4 tags {_N, _V, _J, _X} for
    nounds, verbs, adjectives/adverbs, and others.
    Beware that this method takes a list of tuples and returns a list of strings.
    :param tagged_words: [(str,str)] -- words and their associated POS-tags
    :return: [str] -- words ending with {_N, _V, _J, _X}
    """
    postags = {"N": ["NN", "NNS", "NNP", "NNPS"],
               "V": ["VB", "VBD", "VBG", "VBN", "VBZ", "VBP"],
               "J": ["JJ", "JJR", "JJS"]}
    simplified = []
    for w, t in tagged_words:
        if t in postags["N"]:
            simplified.append("_".join([w, "N"]))
        elif t in postags["V"]:
            simplified.append("_".join([w, "V"]))
        elif t in postags["J"]:
            simplified.append("_".join([w, "J"]))
        else:
            simplified.append("_".join([w, "X"]))
    return simplified

def loop_input(rtype=str, default=None, msg=""):
    """
    Wrapper function for command-line input that specifies an input type and a default value.
    :param rtype: type -- e.g., str, int, float, bool
    :param default: value to be returned if the input is empty
    :param msg: str -- message that is printed as prompt
    :return: value of the specified type
    """
    while True:
        try:
            s = input(msg)
            return rtype(s) if len(s) > 0 else default
        except ValueError:
            print("Input needs to be convertable to",rtype,"-- try again.")
            continue


