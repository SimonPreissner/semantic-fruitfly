import re
import os
import time

import numpy as np
import nltk

from tqdm import tqdm
import utils
import Fruitfly
from Fruitfly import Fruitfly


class Incrementor:

    def __init__(self, corpus_dir, matrix_file,
                 corpus_tokenize=False, corpus_linewise=False, corpus_checkvoc=None,
                 matrix_incremental=True, matrix_maxdims=None, min_count=None, contentwords_only=False,
                 fly_new=False, fly_grow=False, fly_file=None, fly_max_pn=None,
                 verbose=False):
        """
        The purpose of Incrementor objects is to to count co-occurrences of words in text resources and
        maintain a Fruitfly object alongside counting. Both tasks come with several options; most of them
        are realized as attributes of an Incrementor object.
        :param corpus_dir: str -- file path or directory to text resources
        :param matrix_file:  str -- file path (without extension) to the co-occurrence count (e.g., for logging)
        :param corpus_tokenize: bool -- option to tokenize the input text
        :param corpus_linewise: bool -- option to count lines separately (not stable; leave this set to False)
        :param corpus_checkvoc: str -- path to a list of prioritized words (1 word/line) in case of size limitations
        :param matrix_incremental: bool -- use the already existing co-occurrence count in matrix_file
        :param matrix_maxdims: int -- limit the number size of the count (e.g. for faster performance)
        :param min_count: int -- only keep dimensions of words with at least these many occurrences
        :param contentwords_only: bool -- only count tokens tagged with _N,_V,_J, but not _X (requires POS-tagged data)
        :param fly_new: bool -- set up a new Fruitfly object from scratch (only possible with default parameters)
        :param fly_grow: bool -- extend and reduce the Fruitfly's PN layer in parallel to the co-occurrence count
        :param fly_file: str -- file path of the Fruitfly object's config (parameters and connections)
        :param fly_max_pn: int -- limit the number of PNs of the Fruitfly object (doesn't affect matrix_maxdims)
        :param verbose: bool -- comment on current processes via print statements
        :attribute ourcols: str -- file path to the count's vocabulary
        :attribute words: [str] -- text resource as a list of words
        :attribute cooc: ndarray [[]] -- co-occurrence matrix
        :attribute words_to_i: {str:int} -- mapping of the count's vocabulary to the count's dimensions
        :attribute i_to_words: {int:str} -- inverse mapping of words_to_i
        :attribute fruitfly: Fruitfly -- the Fruitfly object to be maintained
        :attribute freq: {str:int} -- frequency distribution of tokens in the current text resource (and earlier ones)
        """
        self.verbose = verbose

        self.corpus_dir   = corpus_dir
        self.is_tokenize  = corpus_tokenize
        self.is_linewise  = corpus_linewise
        self.required_voc = corpus_checkvoc

        self.outspace = matrix_file+".dm"
        self.outcols  = matrix_file+".cols"
        self.is_incremental = matrix_incremental
        self.max_dims = matrix_maxdims
        self.min_count = min_count
        self.postag_simple = contentwords_only

        self.is_new_fly  = fly_new
        self.is_grow_fly = fly_grow
        self.flyfile     = fly_file
        self.fly_max_pn  = fly_max_pn


        self.words = self.read_corpus(self.corpus_dir,
                                      tokenize_corpus=self.is_tokenize,
                                      postag_simple=self.postag_simple,
                                      linewise=self.is_linewise,
                                      verbose=self.verbose)

        self.cooc, self.words_to_i, self.i_to_words, self.fruitfly = \
            self.read_incremental_parts(self.outspace,
                                        self.outcols,
                                        self.flyfile,
                                        verbose=self.verbose)

        # words that will be counted (= labels of the final matrix dimensions)
        self.freq = self.freq_dist(self.words,
                                   size_limit=self.max_dims,
                                   required_words_file=self.required_voc,
                                   verbose=self.verbose)

        if self.verbose: print("\tVocabulary size:",len(self.freq),
                               "\n\tTokens (or lines) for cooccurrence count:",len(self.words))


    #========== FILE READING

    @staticmethod
    def read_corpus(indir, tokenize_corpus=False, postag_simple=False, linewise=False, verbose=False):
        """
        Read text from a file or directory and apply various pre-processing functions.
        Tokenization is optional; set postag_simple=True if the input is POS-tagged with {_N, _V, _J, _X}.
        Returns tokens a single list of strings if linewise==False, or else as list of lists of strings.
        :param indir: str -- a directory or a single file as text resource
        :param tokenize_corpus: bool -- apply nltk.word_tokenize()
        :param postag_simple: bool -- re-uppercase POS-tags
        :param linewise: bool -- not tested; leave this False
        :param verbose: bool -- comment the workings via print statements
        :return: [str] or [[str]] -- depending on the value of linewise
        """
        # this is for initialization of an Incrementor object without resources
        if indir is None:
            if verbose: print("No text resources specified. Continuing with empty corpus.")
            lines = []
        else:
            filepaths = []
            # list of lists of words
            lines = []
            # to delete punctuation entries in simple-POS-tagged data (_N, _V, _J, _X)
            nonword = re.compile("\W+(_X)?")
            # line count
            lc = 0
            # word count
            wc = 0
            # for a single file that is passed
            if os.path.isfile(indir):
                filepaths = [indir]
            else:
                for (dirpath, dirnames, filenames) in os.walk(indir):
                    filepaths.extend([dirpath+"/"+f for f in filenames])

            for file in filepaths:
                try:
                    if verbose: print("reading text from ",file,"...")
                    with open(file) as f:
                        for line in f:
                            lc += 1
                            line = line.rstrip().lower()
                            if tokenize_corpus:
                                tokens = nltk.word_tokenize(line)
                            else:
                                tokens = line.split()
                            linewords = []
                            for t in tokens:
                                # upper-case the POS-tags again
                                if postag_simple:
                                    t = t[:-1]+t[-1].upper()
                                # only appends if the token is not a non-word (=ignores punctuation)
                                if (re.fullmatch(nonword, t) is None):
                                    linewords.append(t)
                                wc+=1
                                if verbose and wc%1000000 == 0:
                                    print("\twords read:",wc/1000000,"million",end="\r")
                            # appends the list of tokens of the current line to the list of lines
                            lines.append(linewords)
                except FileNotFoundError as e:
                    print(e)
            if verbose: print("Finished reading. Number of words:",wc)

        if linewise is False:
            # flattens to a simple word list
            return [w for l in lines for w in l]
        else:
            return(lines)

    def extend_corpus(self, text_resource):
        """
        Takes a file path, reads the file's content, and extends the Incrementor object's available text
        as well as its frequency distribution. Any options and input parameters (e.g., tokenization) are
        handled by the Incrementor object's attributes.
        :param text_resource: file path
        """
        new_text = self.read_corpus(text_resource,
                                    tokenize_corpus=self.is_tokenize,
                                    postag_simple=self.postag_simple,
                                    linewise=self.is_linewise,
                                    verbose=self.verbose)
        self.words.extend(new_text)
        new_freq = self.freq_dist(new_text,
                                  size_limit=self.max_dims,
                                  required_words_file=self.required_voc,
                                  # The following prioritizes old freq keys over new freq keys
                                  required_words=self.freq.keys(),
                                  verbose=self.verbose)
        # update freq with the new counts
        self.freq = self.merge_freqs(self.freq,
                                     new_freq,
                                     required_words_file=self.required_voc,
                                     max_length=self.max_dims)

    def read_incremental_parts(self, outspace, outcols, flyfile, verbose=False):
        """
        Returns a co-occurrence matrix, a corresponding vocabulary and its index, and a Fruitfly object.
        The matrix and the vocabulary can be newly instantiated or taken from existing files.
        The Fruitfly object can be optionally created alongside, also either new or from an
        existing file. All these options are handled by attributes of the Incrementor object
        from which this method is called.
        :param outspace: str -- file path to a co-occurrence count
        :param outcols: str -- file path to the corresponding vocabulary
        :param flyfile: str -- file path to a Fruitfly config (parameters and connections)
        :param verbose: bool -- comment on the workings via print statements
        :return: ndarray [[]] -- co-occurrence matrix (two axes, each of length n)
        :return: {str:int} -- mapping of vocabulary to matrix positions (length: n)
        :return: {int:str} -- mapping of matrix indices to vocabulary (length: n)
        :return: Fruitfly -- Fruitfly object (or None if not wanted)
        """
        if self.is_incremental:
            if verbose: print("\nLoading existing co-occurrence count from",outspace,"...")
            # returns dict of word : vector
            unhashed_space = utils.readDM(outspace)
            i_to_words, words_to_i = utils.readCols(outcols)
            dimensions = sorted(words_to_i, key=words_to_i.get)
            cooc = np.stack(tuple([unhashed_space[w] for w in dimensions]))
        else:
            cooc = np.array([[]])
            words_to_i = {}
            i_to_words = {}

        if self.is_grow_fly:
            if self.is_new_fly:
                if verbose: print("creating new fruitfly...")
                # default config: (50,40000,6,5,log)
                fruitfly = Fruitfly.from_scratch(max_pn_size=self.fly_max_pn)
            else:
                if verbose: print("loading fruitfly from",flyfile,"...")
                fruitfly = Fruitfly.from_config(flyfile)
                self.fly_max_pn = fruitfly.max_pn_size
        else:
            fruitfly = None

        return cooc, words_to_i, i_to_words, fruitfly

    def freq_dist(self, wordlist, size_limit=None, required_words_file=None, required_words=None, verbose=False):
        """
        This method is used to limit the dimensionality of the count matrix, which speeds up processing.
        The obtained dictionary is used as vocabulary reference of the current corpus at several processing steps.
        For true incrementality, size_limit is None and the dictionary is computed over the currently available corpus.
        If size_limit is None, required_words has no effect on the obtained dictionary.
        :param wordlist: [str] -- list of (word) tokens from the text resource
        :param size_limit: int -- maximum length of the returned frequency distribution
        :param required_words_file: str -- file path to a list with prioritized words (regardless of their frequencies)
        :param required_words: [str] -- used to pass already existing freq keys if freq needs to be extended
        :param verbose: bool -- comment on workings via print statements
        :return: {str:int} -- frequency distribution
        """
        if verbose: print("creating frequency distribution over",len(wordlist),"tokens...")
        freq = {}
        # the linewise option is not tested.
        if self.is_linewise:
            for line in tqdm(wordlist):
                for w in line:
                    if self.postag_simple:
                        # only counts nouns, verbs, and adjectives/adverbs
                        if w.endswith(("_N", "_V", "_J")):
                            if w in freq:
                                freq[w] += 1
                            else:
                                freq[w] = 1
                    else:
                        if w in freq:
                            freq[w] += 1
                        else:
                            freq[w] = 1
        else:
            for w in tqdm(wordlist):
                if self.postag_simple:
                    if w.endswith(("_N", "_V", "_J")):
                        if w in freq:
                            freq[w] += 1
                        else:
                            freq[w] = 1
                else:
                    if w in freq:
                        freq[w] += 1
                    else:
                        freq[w] = 1

        # list of all words
        frequency_sorted = sorted(freq, key=freq.get, reverse=True)
        # find out the required words
        if required_words_file is None and required_words is None:
            returnlist = frequency_sorted
        else:
            # required words can be passed as [str] or as file path, and both at the same time is possible
            checklist = []
            if required_words_file is not None:
                checklist.extend(self.read_checklist(required_words_file))
            if required_words is not None:
                checklist.extend(required_words)

            overlap = list(set(checklist).intersection(set(frequency_sorted)))
            # words that are not required; sorted by frequency
            rest_words = [w for w in frequency_sorted if w not in overlap]
            returnlist = overlap+rest_words
        # impose a size limit if wanted
        if(size_limit is not None and size_limit <= len(freq)):
            return {k:freq[k] for k in returnlist[:size_limit]}
        else:
            return freq

    def merge_freqs(self, freq1, freq2, required_words_file=None, max_length=None):
        """
        Merges two frequency distributions while preserving the possible required words and
        a possible length restriction. As for the returned frequency distribution,
        length restriction has a higher priority than required words and more frequent words
        have a higher priority to be returned than less frequent words.
        :param freq1: {str:i} -- frequency distribution over word tokens
        :param freq2: [str:i} -- frequency distribution over word tokens
        :param required_words_file: str -- file path to a word list (one word per line)
        :param max_length: int
        :return: {str:i} -- merged frequency distribution, possibly limited in length
        """
        for k, v in freq2.items():
            if k in freq1:
                freq1[k] += v
            else:
                freq1[k] = v

        # list of all words, sorted by frequency
        frequency_sorted = sorted(freq1, key=freq1.get, reverse=True)

        if required_words_file is None:
            returnlist = frequency_sorted
        else:
            checklist = self.read_checklist(required_words_file)
            # still a frequency-sorted word list
            overlap = [w for w in frequency_sorted if w in checklist]
            # words that are not required; sorted by frequency
            rest_words = [w for w in frequency_sorted if w not in overlap]
            returnlist = overlap + rest_words

        if (max_length is not None and len(freq1) > max_length):
            return {k: freq1[k] for k in returnlist[:max_length]}
        else:
            return freq1

    @staticmethod
    def read_checklist(checklist_filepath):
        """
        Reads a file, but continues with empty checklist if the file cannot be found.
        :param checklist_filepath: str -- file with one word per line
        :return: [str] -- words to be checked for overlap
        """
        try:
            checklist = []
            with open(checklist_filepath, "r") as f:
                for word in f:
                    word = word.rstrip()
                    checklist.append(word)
            return checklist
        except FileNotFoundError:
            if checklist_filepath is None:
                return []
            else:
                print("Checklist file path not found. Continuing without checklist.")
                return []

    def check_overlap(self, checklist_filepath=None, wordlist=None):
        """
        Compares two lists of words (one from a file and one internal list) for complete overlap.
        :param checklist_filepath: str -- file should contain oe word per line
        :param wordlist: [str] -- default value: the Incrementor object's vocabulary
        :return: bool -- whether there is a complete overlap
        :return: [str] -- words that are in checklist but not in wordlist
        """
        if self.verbose: print("\nchecking overlap...")
        if checklist_filepath is None: checklist_filepath = self.required_voc
        if wordlist is None: wordlist = self.freq.keys()
        checklist = self.read_checklist(checklist_filepath)

        if len(checklist) == 0:
            if self.verbose: print("\tcheck_overlap(): nothing to check.")
            return True, []

        unshared_words = list(set(checklist).difference(set(wordlist)))

        if self.verbose:
            if len(unshared_words) == 0:
                print("\tComplete overlap with",checklist_filepath)
            else:
                print("\tChecked for overlap with",checklist_filepath,
                      "\n\twords missing in the corpus:",len(unshared_words),
                      "\n\texamples:",unshared_words[:10])

        return len(unshared_words) == 0, unshared_words


    #========== CO-OCCURRENCE COUNTING

    def extend_incremental_parts_if_necessary(self, w):
        """
        If a new (i.e. not in Incrementor.words_to_i) word is passed, this adds a new dimension along both axes
        of the count matrix and, if a Fruitfly object is maintained, calls the Fruitfly.py extension mechanism.
        This method is not optimized for efficiency and slows down the counting process.
        :param w: str -- word to be checked for necessity to extend
        """
        if w not in self.words_to_i:
            # extend the vocabulary and index
            self.words_to_i[w] = len(self.words_to_i)
            self.i_to_words[self.words_to_i[w]] = w
            # make bigger matrix
            temp = np.zeros((len(self.words_to_i), len(self.words_to_i)))
            # paste current matrix into the new one
            temp[0:self.cooc.shape[0], 0:self.cooc.shape[1]] = self.cooc
            self.cooc = temp
            # PN layer extension only needed if count size is greater than PN layer size
            if self.fruitfly is not None and len(self.words_to_i) > self.fruitfly.pn_size:
                self.fruitfly.extend()

    def count_start_of_text(self, words, window): # for the first couple of words
        """
        Counts co-occurrences of the first words of a text in the same fashion
        as count_middle_of_text and count_end_of_text.
        Only counts words that are also in self.freq.
        If a new word is encountered, the extension mechanism is called.
        :param words: [str]
        :param window: int -- window to one side
        """
        # iterate over the first words
        for i in range(window):
            if words[i] in self.freq:
                # iterate over the context
                for c in range(i+window+1):
                    if words[c] in self.freq:
                        self.extend_incremental_parts_if_necessary(words[i])
                        self.extend_incremental_parts_if_necessary(words[c])
                        self.cooc[self.words_to_i[words[i]]][self.words_to_i[words[c]]] += 1
                # delete "self-occurrence"
                self.cooc[self.words_to_i[words[i]]][self.words_to_i[words[i]]]-=1

    def count_middle_of_text(self, words, window):
        """
        Counts co-occurrences of most of the words of a text.
        The counting was split up for better performance and in order to avoid index errors.
        Only counts words that are also in self.freq.
        If a new word is encountered, the extension mechanism is called.
        :param words: [str]
        :param window: int -- window to one side
        """
        # this part is without tqdm; the other one is with.
        if self.is_linewise:
            # iterate over words (of a line)
            for i in range(window, len(words)-window):
                if words[i] in self.freq:
                    # iterate over the context
                    for c in range(i-window, i+window+1):
                        if words[c] in self.freq:
                            self.extend_incremental_parts_if_necessary(words[i])
                            self.extend_incremental_parts_if_necessary(words[c])
                            self.cooc[self.words_to_i[words[i]]][self.words_to_i[words[c]]] += 1
                    # delete "self-occurrence"
                    self.cooc[self.words_to_i[words[i]]][self.words_to_i[words[i]]]-=1
        # count assuming that the whole text is passed -> tqdm is feasible
        else:
            # iterate over words
            for i in tqdm(range(window, len(words)-window)):
                if words[i] in self.freq:
                    # iterate over the context
                    for c in range(i-window, i+window+1):
                        if words[c] in self.freq:
                            self.extend_incremental_parts_if_necessary(words[i])
                            self.extend_incremental_parts_if_necessary(words[c])
                            self.cooc[self.words_to_i[words[i]]][self.words_to_i[words[c]]] += 1
                    # delete "self-occurrence"
                    self.cooc[self.words_to_i[words[i]]][self.words_to_i[words[i]]]-=1

    def count_end_of_text(self, words, window):
        """
        Counts co-occurrences of the last words of a text in the same fashion
        as count_start_of_text and count_middle_of_text.
        Only counts words that are also in self.freq.
        If a new word is encountered, the extension mechanism is called.
        :param words: [str]
        :param window: int -- window to one side
        """
        # iterate over the last words
        for i in range(len(words)-window, len(words)):
            if words[i] in self.freq:
                # iterate over the context
                for c in range(i-window, len(words)):
                    if words[c] in self.freq:
                        self.extend_incremental_parts_if_necessary(words[i])
                        self.extend_incremental_parts_if_necessary(words[c])
                        self.cooc[self.words_to_i[words[i]]][self.words_to_i[words[c]]] += 1
                # delete "self-occurrence"
                self.cooc[self.words_to_i[words[i]]][self.words_to_i[words[i]]]-=1

    def count_cooccurrences(self, words=None, window=5, timed=False):
        """
        Wraps up the three methods which count co-occurrences and comments on
        the progress if Incrementor.verbose==True.
        :param words: list[str] -- tokens to be counted
        :param window: int -- specifies window size to one side.
        :param timed: bool -- If True, this method returns the time taken to execute the method
        :return: float -- Seconds taken to execute the method
        """
        t0 = time.time()
        # allow counting of not every word (e.g. of only content words)
        if words is None: words = self.words
        if self.verbose: print("\ncounting co-occurrences within",window,"words distance...")
        if self.is_linewise:
            for line in tqdm(words):
                # to avoid index errors
                if len(line) >= 2*window:
                    self.count_start_of_text(line, window)
                    self.count_middle_of_text(line, window)
                    self.count_end_of_text(line, window)
                else:
                    if self.verbose: print("\tline too short for cooccurrence counting:",line)
        else:
            self.count_start_of_text(words, window)
            self.count_middle_of_text(words, window)
            self.count_end_of_text(words, window)
        if self.verbose: print("finished counting; matrix shape:",self.cooc.shape)

        if timed is True:
            return time.time()-t0
        else:
            pass

    def reduce_count_size(self, verbose=False, timed=False):
        """
        Uses the Incrementor object's frequency distribution (over the currently processed text)
        to delete the dimensions of words that occur less often than a given threshold. It also
        makes a call to the Fruitfly object to reduce the PN layer, if needed.
        This is costly, but preserves incrementality and is cognitively plausible by acting like
        a rudimentary attention mechanism.
        :return: int -- number of words that are deleted by reducing
        :return: float -- time taken to execute this method
        """
        t0 = time.time()
        if self.min_count is None:
            if timed:
                return 0, time.time()-t0
            else:
                return 0
        else:
            # freq (current words) and words_to_i (all words encountered so far) might differ
            counted_freq_words = set(self.words_to_i).intersection(set(self.freq))
            if verbose: print("Deleting infrequent words (less than",self.min_count+1,"occurrences) from the count matrix...")
            delete_these_w = [w for w in counted_freq_words if self.freq[w]<=self.min_count]
            delete_these_i = [self.words_to_i[w] for w in delete_these_w]
            # delete rows and columns from the count matrix
            self.cooc = np.delete(self.cooc, delete_these_i, axis=0)
            self.cooc = np.delete(self.cooc, delete_these_i, axis=1)
            # delete elements from the dictionary
            for w in delete_these_w:
                del(self.words_to_i[w])
            # in the index, shift words from higher dimensions to the freed-up dimensions
            self.i_to_words = {i:w for i,w in enumerate(sorted(self.words_to_i, key=self.words_to_i.get))}
            # update the index mapping in the dictionary
            self.words_to_i = {w:i for i,w in self.i_to_words.items()}
            # also reduce the PN layer of the Fruitfly!
            if self.fruitfly is not None and self.fruitfly.pn_size > self.cooc.shape[0]:
                self.fruitfly.reduce_pn_layer(delete_these_i, self.cooc.shape[0])

            if verbose: print("\t",len(delete_these_w),"words deleted. New count dimensions:",self.cooc.shape)
            if timed:
                return len(delete_these_w), time.time()-t0
            else:
                return len(delete_these_w)


    #========== LOGGING

    def log_matrix(self, outspace=None, outcols=None, only_these=None):
        """
        Writes the Incrementor object's count matrix and the corresponding dictionary
        to a file each. It is also possible to only log a certain selection of vectors.
        Without parameter specification, the Incrementor object's variables are used.
        :param outspace: str -- file path for the count to be logged
        :param outcols: str -- file path for the count's vocabulary to be logged
        :param only_these: {str:int} words and their indices (subset of words_to_i)
        """
        # optional parameters allow external use
        if outspace is None: outspace = self.outspace
        if outcols  is None: outcols = self.outcols
        if only_these is None: only_these = self.words_to_i

        with open(outspace, "w") as dm_file, open(outcols, "w") as cols_file:
            if self.verbose: print("\nwriting vectors to",outspace,"\nwriting dictionary to",outcols,"...")
            # sorted by index
            for word,i in tqdm(sorted(only_these.items(), key=lambda x: x[1])):
                # line number represents index in the cols file
                cols_file.write(word+"\n")
                vectorstring = " ".join([str(v) for v in self.cooc[i]])
                dm_file.write(word+" "+vectorstring+"\n")

    def log_fly(self, flyfile=None):
        """
        This is a wrapper for Fruitfly.log_params().
        By default, it logs to the Incrementor object's 'flyfile' attribute
        :param flyfile: str -- file path for the fly to be logged
        """
        if flyfile is None: flyfile = self.flyfile
        if flyfile is not None and self.fruitfly is not None:
            if self.verbose: print("logging fruitfly to",flyfile,"...")
            self.fruitfly.log_params(filename=flyfile)

    def get_setup(self):
        """
        Get a dictionary with all parameters and attributes of the Incrementor object.
        Returns a small extract from those attributes with larger data structures.
        :return: {str:object}
        """
        return {
            "verbose":self.verbose,
            "corpus_dir":self.corpus_dir,
            "is_tokenize":self.is_tokenize,
            "is_linewise":self.is_linewise,
            "required_voc":self.required_voc,
            "outspace":self.outspace,
            "outcols":self.outcols,
            "is_incremental":self.is_incremental,
            "max_dims":self.max_dims,
            "min_count":self.min_count,
            "postag_simple":self.postag_simple,
            "is_new_fly":self.is_new_fly,
            "is_grow_fly":self.is_grow_fly,
            "flyfile":self.flyfile,
            "fly_max_pn":self.fly_max_pn,
            "cooc":self.cooc[:10,:10],
            "words_to_i":sorted(self.words_to_i, key=self.words_to_i.get)[:10],
            "i_to_words":sorted(self.i_to_words)[:10],
            "fruitfly":self.fruitfly.get_specs(),
            "words":self.words[:20],
            "freq":sorted(self.freq, key=self.freq.get, reverse=True)[:10]
        }




if __name__ == '__main__':
    """
    This working example of the Incrementor class
    reads in a text resource, counts co-occurrences,
    optionally maintains a Fruitfly object alongside,
    and logs the resulting parts.
    The FFA is not applied here.    
    """

    # File input
    infile = utils.loop_input(rtype=str, default="data/chunks_wiki",
                              msg="Path to text resources (default: data/chunks_wiki): ")
    outfiles = utils.loop_input(rtype=str, default="data/count",
                                msg="Path/name of the output count data (without extension; default: data/count): ")
    xvoc = utils.loop_input(rtype=str, default=None, msg="Path to a word list to be checked for overlap (optional): ")

    # Parameters for counting process
    tknz = False if input("Tokenize the input text? (default: yes) [y/n]").upper() == "N" else True
    incr = True if input("Work incrementally (= use count data as basis) (default: no)? [y/n]: ").upper() == "Y" else False
    window = utils.loop_input(rtype=int, default=5, msg="Window size (to each side) for counting (default: 5): ")
    dims = utils.loop_input(rtype=int, default=None,
                            msg="Maximum vocabulary size for the count (skip this for true incrementality): ")
    minc = utils.loop_input(rtype=int, default=None,
                        msg="Periodic deletion of words with n occurrences or fewer from the count (optional) -- n: ")

    # FFA parameter input
    grow = utils.loop_input(rtype=bool, default=False,
                            msg="Maintain an FFA object alongside counting (default: no)? [y/n]: ")
    if grow:
        nfly = True if input("Make a new standard FFA (default: no)? [y/n]: ").upper() == "Y" else False
        if nfly:
            fcfg = utils.loop_input(rtype=str, default="data/fly_config", msg="File path of the new FFA's config: ")
        else:
            fcfg = utils.loop_input(rtype=str, default="data/fly_config", msg="File path of the existing FFA config: ")
    else:
        nfly = None
        fcfg = None
    is_verbose = False if input("Be verbose while running? (default: yes) [y/n]").upper() == "N" else True

    # working code
    incrementor = Incrementor(infile, outfiles,
                              corpus_tokenize=tknz, corpus_linewise=False, corpus_checkvoc=xvoc,
                              matrix_incremental=incr, matrix_maxdims=dims, min_count=minc,
                              fly_new=nfly, fly_grow=grow, fly_file=fcfg, fly_max_pn=None,
                              verbose=is_verbose)
    # checking for overlap is not really necessary, but might be informative
    all_in, unshared_words = incrementor.check_overlap(checklist_filepath=xvoc)
    incrementor.count_cooccurrences(words=incrementor.words, window=window)
    incrementor.log_matrix()
    incrementor.log_fly()

    if is_verbose: print("done.")


