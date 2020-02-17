import time
import numpy as np
from math import ceil
from tqdm import tqdm


class Fruitfly:
    """
    This class contains all the architecture and methods of the FFA:
    - PN layer, KC layer, projection connections
    - input flattening, projection, hashing
    - output and configuration logging
    - dynamic growth and reduction of PN layer and connections
    It does not implement any co-occurrence counting or evaluation.
    """

#========== CONSTRUCTORS AND SETUP

    def __init__(self, pn_size, kc_size, proj_size, hash_percent, flattening, max_pn_size=None, old_proj=None):
        """
        Creates layers and random projections. For initialization, use one of the class methods.
        :param pn_size: int -- (initial) size of the PN layer (PN = Projection Neuron)
        :param kc_size: int -- size of the KC layer (fixed) (KC = Kenyon Cell)
        :param proj_size: int -- number of connections per KC
        :param hash_percent: int -- percentage of winners in the WTA procedure
        :param flattening: str -- function by which to counter-act the natural word frequency effects
        :param max_pn_size: int -- size limit of the PN layer
        :param old_proj: {int:[int]} mapping of KC index to PN indices (if constructing from a file)
        :attribute proj_functions: {int:[int]} mapping of KC index to PN indices ("incoming connections")
        :attribute pn_to_kc: {int:[int]} mapping of PN index to KC indices (i.e. inverse mapping of proj_functions)
        """

        self.flattening = flattening
        if self.flattening not in ["log", "log2", "log10"]:
            print("No valid flattening method for the FFA. Continuing without flattening.")
        self.pn_size = pn_size
        self.kc_size = kc_size
        self.kc_factor = kc_size/pn_size
        self.proj_size = proj_size
        self.hash_percent = hash_percent
        self.max_pn_size = max_pn_size

        self.pn_layer = np.zeros(self.pn_size)
        self.kc_layer = np.zeros(self.kc_size)

        # arrays of PNs that are connected to any one KC 
        self.proj_functions = old_proj if old_proj is not None else self.create_projections()
        self.pn_to_kc = self.forward_connections([i for i in range(self.pn_size)])

    @classmethod
    def from_config(cls, filename):
        """
        Reads parameters from a file that starts with the parameters (1 per line, name folloewd by value),
        followed by the connections (KC index followed by its connected PN indices, sparated by spaces,
        1 KC per line). The actual initialization is executed in the constructor, which is called upon return.
        :param filename: file path to configuration file
        :return: call to the constructor
        """
        try:
            with open(filename, "r") as f:
                lines = f.readlines()
            specs = {}

            lnr = 0
            # marks beginning of conncetions
            con_ind = 0

            # read in parameters
            paramline = True
            while paramline:
                # first element of params contains a string if parameter, else an int (if connections)
                params = lines[lnr].rstrip().split()
                # only connection lines have int-able first elements
                try:
                    int(params[0])
                    paramline = False
                    con_ind = lnr
                except ValueError:
                    # treats line as containing a parameter (and not as containing connections)
                    try:
                        specs[params[0]]=int(params[1])
                    except ValueError:
                        # leaves string parameters (e.g. flattening) as strings
                        specs[params[0]]=params[1]
                    lnr+=1

            if "max_pn_size" not in specs or specs["max_pn_size"] == "None":
                specs["max_pn_size"] = None

            # read in connections
            connections = {}
            for line in lines[con_ind:]:
                values = line.split()
                # makes a mapping of {kc:[pn]}
                connections[int(values[0])] = [int(v) for v in values[1:]]

            return cls(specs["pn_size"], specs["kc_size"], specs["proj_size"], specs["hash_perc"],
                       specs["flattening"], max_pn_size=specs["max_pn_size"], old_proj=connections)
        except FileNotFoundError:
            print("FileNotFoundError in Fruitfly.from_config()!\n"
                  "\tcontinuing with a default Fruitfly object (50, 40000, 6, 5, log)!")
            return Fruitfly.from_scratch()

    @classmethod
    def from_scratch(cls, pn_size=50, kc_size=40000, proj_size=6, hash_percent=5, flattening="log", max_pn_size=None):
        """
        This is a workaround for issues with the default constructor.
        :param pn_size: int -- (initial) size of the PN layer (PN = Projection Neuron)
        :param kc_size: int -- size of the KC layer (fixed) (KC = Kenyon Cell)
        :param proj_size: int -- number of connections per KC
        :param hash_percent: int -- percentage of winners in the WTA procedure
        :param flattening: str -- function by which to counter-act the natural word frequency effects
        :param max_pn_size: int -- size limit of the PN layer"""
        return cls(pn_size, kc_size, proj_size, hash_percent, flattening, max_pn_size=max_pn_size)

    def create_projections(self):
        """
        Creates random connections between the PN layer and the KC layer.
        :return: {int:ndarray[int]} -- mapping of KC to the PNs that are connected to it
        """
        proj_functions = {}
        print("\nCreating new projections...")

        for cell in tqdm(range(self.kc_size)):
            # uniform random choice + maximally 1 connection per PN-KC pair
            activated_pns = list(set(np.random.randint(self.pn_size, size=self.proj_size)))
            proj_functions[cell] = activated_pns

        return proj_functions

    def forward_connections(self, pn_indices):
        """
        For the passed PNs, returns a mapping to the KCs to which each PN connects.
        The retured mapping is a subset of the inverse of the mapping returned by create_projections().
        :param pn_indices: int or [int] -- PN indices
        :return: {int:[int]} -- mapping of PN index to connected KC indices
        """
        pn_indices = [pn_indices] if type(pn_indices) != list else pn_indices

        # { pn_index : [connected KCs] }
        pn_to_kc = {pn:[] for pn in pn_indices}
        for kc,connections in self.proj_functions.items():
            # only for the PNs given to the method!
            for pn in pn_indices:
                if pn in connections:
                    pn_to_kc[pn].append(kc)

        return pn_to_kc

#========== STRINGS AND LOGGING
            
    def show_off(self):
        """
        Return the Fruitfly algorithm parameter values as string. Useful for command line output and logging.
        :return: str
        """
        statement = "pn_size\t"     + str(self.pn_size)+"\n"+\
                    "kc_factor\t"   + str(self.kc_factor)+"\n"+\
                    "kc_size\t"     + str(self.kc_size)+"\n"+\
                    "proj_size\t"   + str(self.proj_size)+"\n"+\
                    "hash_perc\t"   + str(self.hash_percent)+"\n"+ \
                    "flattening\t"  + str(self.flattening) + "\n" + \
                    "max_pn_size\t" + str(self.max_pn_size)
        return statement

    def get_specs(self):
        """
        Return the Fruitfly algorithm parameter values as dictionary. Useful for in-code usage
        :return: {str:int or str} -- keys are the same as class attribute names.
        """
        return {"pn_size":self.pn_size,
                "kc_factor":self.kc_factor, 
                "kc_size":self.kc_size,
                "proj_size":self.proj_size, 
                "hash_percent":self.hash_percent,
                "flattening": self.flattening,
                "max_pn_size":self.max_pn_size}

    def log_params(self, filename="log/configs/ff_config.cfg", timestamp=False):
        """
        Writes parameters and projection connections to a specified file.
        :param filename: str -- file path, should have the extension .cfg
        :param timestamp: bool -- optionally include a time stamp in the file name
        """
        if timestamp is True:
            filename = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())+"_"+filename
        connections = ""
        print("Logging fruitfly config to",filename,"...")
        for kc,pns in tqdm(self.proj_functions.items()):
            # connections logged as: kc pn pn pn pn...; 1 KC per line
            connections+=(str(kc)+" "+" ".join([str(pn) for pn in pns])+"\n")
        with open(filename, "w") as logfile:
            # first log the parameters
            logfile.write(self.show_off()+"\n")
            # then log the connections
            logfile.write(connections)

    def important_words_for(self, word_hash, pn_dic, n=None):
        """ 
        For every PN that is connected to an activated KC of the given hash, count the number of connections
        from that PN to KCs that are activated in that hash. The more often an active KC is connected to a
        certain PN, the more important will that PN be (and subsequently, the word which is represented by the PN)
        :param word_hash: [int] -- binary hash signature
        :param pn_dic: {int:str} -- mapping of PN index to the word that it represents
        :param n: int -- number of words to be returned
        :return: [str] -- words whose PNs were important during FFA application; sorted (descending) by importance
        """
        if len(pn_dic) != self.pn_size:
            print("WARNING: in Fruitfly.important_words_for(): \
            vocabulary doesn't match PN layer!", end=" ")
            print("Make sure to call this method with the vocabulary obtained from flying! \
            Continuing with 'wrong' vocabulary")
        important_words = {} # dict of word:count_of_connections
        for i in range(len(word_hash)):
            # only look at PNs of activated KCs
            if int(word_hash[i]) == 1:
                # retrieve transitions of the activated KC
                activated_pns = self.proj_functions[i]
                # and iterate over those PNs
                for pn in activated_pns:
                    # retrieve word from PN index
                    w = pn_dic[pn]
                    if w in important_words:
                        important_words[w]+=1 
                    else:
                        important_words[w]=1

        #TODO this could be changed to return important_words itself to also return a value of importance for a word
        count_ranked = sorted(important_words, key=important_words.get, reverse=True)
        if n is None: 
            return count_ranked
        else:
            # only return the n most important words
            return count_ranked[:n]

#========== INCREMENTALITY

    def extend(self):
        """
        Extends the Fruitfly object's PN layer and connects it to the KC layer by either making
        new connections to KCs that have not yet reached the limit of connections or, if a KC is
        "full", by reallocating existing connections from other PNs.
        The choice of KCs to which the new PN is connected is random, but biased towards less
        often connected KCs. This ensures an even distribution of the connections coming into the
        KC layer.
        The reallocation mechanism chooses connected PNs for the "stealing" of connections with a
        priority on towards PNs with an above-average number of outgoing connections.
        This ensures an even distribution of the connections going from the PN layer.
        """
        # don't extend if there's a limit.
        if self.max_pn_size is not None and self.pn_size == self.max_pn_size:
            return

        # add a PN to the Fruitfly and change its parameters accordingly
        self.pn_size+=1
        self.pn_layer = np.append(self.pn_layer, [0])
        self.kc_factor = self.kc_size/self.pn_size
        # number of connections from the new PN = avg. PN connectedness
        new_avg_pn_con = int(sum([len(p) for k,p in self.proj_functions.items()])/self.pn_size)

        weighted_kcs = {}
        for cell in self.proj_functions:
            # weight the KC with the inverse of its connectedness
            weighted_kcs[cell] = 1.0/(1+len(self.proj_functions[cell]))
            weighted_kcs[cell] = weighted_kcs[cell]*np.random.rand()
        # these winners connect to the new PN
        winners = sorted(weighted_kcs, key=weighted_kcs.get, reverse=True)[:new_avg_pn_con]

        # add PN to connections of the winner KCs
        for kc in winners:
            # fully connected winner KCs experience connection switching
            if len(self.proj_functions[kc]) == self.proj_size:
                pn_con = {pn:len(self.pn_to_kc[pn]) for pn in self.proj_functions[kc]}
                # the most connected of the PNs for this winner KC gets robbed
                robbed_pn = sorted(pn_con, key=pn_con.get, reverse=True)[0]
                # replace PN indices in proj_functions
                self.proj_functions[kc][self.proj_functions[kc].index(robbed_pn)] = self.pn_size-1
                # update pn_to_kc
                del self.pn_to_kc[robbed_pn][self.pn_to_kc[robbed_pn].index(kc)]

            # non-full KCs receive a new connection
            else:
                self.proj_functions[kc].append(self.pn_size-1)

        self.pn_to_kc.update(self.forward_connections([self.pn_size-1]))

    def reduce_pn_layer(self, del_indices, new_pn_size):
        """
        When a Fruitfly object is maintained parallelly to a co-occurrence count,
        that count might be 'pruned' and thus dimensions might be deleted.
        This method reduces the PN layer by the PNs of those words that are deleted
        from the co-occurrence count.
        This entails a change of the mapping of vocabulary to PN layer, of the mappings
        between PN layer KC layer, and a freeing-up of KC connections.
        :param del_indices: [int] -- the positions that are deleted from the count
        :param new_pn_size: int -- usually the size of the count matrix (in order to fit the PN layer to the count)
        """
        # make a mapping that represents the shift induced by deleting PNs (important to keep the correct connections)
        old_to_new_i = {}
        newi = 0
        for oldi in range(self.pn_size):
            if oldi in del_indices:
                pass
            else:
                old_to_new_i[oldi] = newi
                newi += 1
        # The KC layer is independent from pn_size and can be modified before the other components
        for kc,pns in self.proj_functions.items():
            # choose remaining PNs and do a look-up in the mapping for shifted PNs
            self.proj_functions[kc] = [old_to_new_i[oldi] for oldi in list(set(pns).difference(set(del_indices)))]
        # update the pn_layer to be of same size as the count matrix
        self.pn_size = new_pn_size
        # re-size the PN layer
        self.pn_layer = np.zeros(self.pn_size)
        # re-do the forward connections
        self.pn_to_kc = self.forward_connections([i for i in range(self.pn_size)])

    def fit_space(self, unhashed_space, words_to_i):
        """
        Returns vectors which fit (number of dimensions) to pn_size. This enables the Incrementor
        to count independently from the Fruitfly. this method also pads dimensions if the vectors
        are shorter than pn_size.
        Dimension reduction deletes the words from the vectors which have co-occurred the least
        often with words in the space. The dimensions of the returned space are sorted alphabetically.
        :param unhashed_space: {str:[float]} -- words and their corresponding co-occurrence counts
        :param words_to_i: {str:int} -- mapping of vocabulary to the dimension in the co-occurrence count
        :return: {str:[int]} -- co-occurrence counts of length pn_size
        :return: {str:int} -- mapping of vocabulary to PN index (and to the returned vectors)
        :return: {int:str} -- inverse mapping: PN indices to vocabulary
        """

        # pad the vectors if they haven't reached pn_size yet (only in early stages)
        if len(words_to_i) < self.pn_size:
            print("unhashed_space needs to be padded:",len(unhashed_space),"to",self.pn_size,"dimensions.")
            pad_size = self.pn_size - len(words_to_i)
            padded_space = {w:np.append(vec, np.zeros(pad_size)) for w,vec in unhashed_space.items()}
            padded_dic = {w:i+pad_size for w,i in words_to_i.items()}
            padded_ind = {v:k for k,v in padded_dic.items()}
            return padded_space, padded_dic, padded_ind

        # max_pn_size not defined or not yet reached --> fitting not needed
        elif self.max_pn_size is None or len(words_to_i)<=self.max_pn_size:
            return unhashed_space, words_to_i, {v: k for k, v in words_to_i.items()}

        # the space has more dimensions than pn_size --> fitting needed
        else:
            # extract the most frequent words
            vecsums = np.zeros(len(unhashed_space[list(unhashed_space.keys())[0]]))
            for w,vec in unhashed_space.items():
                vecsums += vec
            freq = {w:vecsums[i] for w,i in words_to_i.items()}
            # only keep the most frequent context words
            new_keys = sorted(freq, key=freq.get, reverse=True)[:self.max_pn_size]
            fitted_space = {}
            old_dims = [i for w,i in words_to_i.items() if w not in new_keys]
            for w,vec in unhashed_space.items():
                fitted_space[w] = np.delete(vec,old_dims)
            # sort words alphabetically (this sorts the space)
            new_keys.sort()
            new_dic = {k:new_keys.index(k) for k in new_keys}
            new_ind = {v:k for k,v in new_dic.items()}

            return fitted_space, new_dic, new_ind


#========== FFA APPLICATION

    def flatten(self, frequency_vector):
        """ 
        Counteracts the Zipfian distribution of words (which leads to very unequal co-occurrence
        values) by applying log, log2, or log10 to each count of a given vector. The flattening
        function is specified during initilization of the Fruitfly object.
        :return: [float] -- 'flattened' vector
        """
        flat_vector = np.zeros(len(frequency_vector))

        if self.flattening == "log":
            for i, freq in enumerate(frequency_vector):
                # '1.0+' for co-occurrence values of 0
                flat_vector[i] = np.log(1.0+freq)
        elif self.flattening == "log2":
            for i, freq in enumerate(frequency_vector):
                flat_vector[i] = np.log2(1.0+freq)
        elif self.flattening == "log10":
            for i, freq in enumerate(frequency_vector):
                flat_vector[i] = np.log10(1.0+freq)
        else: 
            return frequency_vector
        return flat_vector

    def projection(self):
        """
        For each KC, sum up the values of the PNs that have a connection to this KC. Return the sums
        ( = activated KC layer).
        :return: [float] -- activated KC layer
        """
        kc_layer = np.zeros(self.kc_size)
        for cell in range(self.kc_size):
            # PNs connected to this particular KC
            activated_pns = self.proj_functions[cell]
            for pn in activated_pns:
                kc_layer[cell]+=self.pn_layer[pn]
        return kc_layer

    def hash_kenyon(self):
        """
        Choose the most activated KCs, set them to 1 and the rest to 0
        :return: [int] -- binary array; = hashed vector
        """
        kc_activations = np.zeros(self.kc_size)
        # number of winners
        top = int(ceil(self.hash_percent * self.kc_size / 100))
        # select those KCs with the highest activation
        activated_kcs = np.argpartition(self.kc_layer, -top)[-top:]
        for cell in activated_kcs:
            kc_activations[cell] = 1
        return kc_activations

    def fly(self, unhashed_space, words_to_i, timed=False):
        """
        Hash each element of the input space. 
        Fit input space to the Fruitfly's PN layer (if necessary) by
        choosing the most frequent words as dimensions. Afterwards, apply 
        flattening before input, afterwards project, hash, and return the 
        complete hashed space.
        :param unhashed_space: {str:[int]} -- words and their raw co-ocurrence counts
        :param words_to_i: {str:int} -- mapping of vocabulary to dimension in the co-occurrence count
        :param timed: bool -- optionally return the time taken for execution
        :return: {str:[int]} -- words and their binary hash signatures
        :return: {str:int} -- mapping of words (from unhashed_space) to indices (of the PN layer)
        :return: {int:str} -- inverse mapping: PN indices to words of the dimensions used for hashing
        :return: float -- time taken for execution
        """
        t0 = time.time()
        print("Starting flying...")
        # choose the most frequent words in the count as dimensions for the PN layer
        fitted_space, flight_dic, flight_ind = self.fit_space(unhashed_space, words_to_i)

        space_hashed = {}
        # hashes count vectors one by one
        for w in tqdm(fitted_space):
            # initialize the PN activation for this vector
            self.pn_layer = self.flatten(fitted_space[w])
            # sum PN activations to obtain the KC activations
            self.kc_layer = self.projection()
            # WTA procedure; hashes have the same dimensionality as kc_layer
            space_hashed[w] = self.hash_kenyon()
        if timed is True:
            return space_hashed, flight_dic, flight_ind, time.time()-t0
        else:
            return space_hashed, flight_dic, flight_ind


if __name__ == '__main__':
    """
    This working example of the application of the FFA reads in a space (= words and vectors) and evaluates
    on a given test set by means of Spearman Correlation:
    it either applies the FFA (one or multiple times) and evaluates the improvement of the hashes over the unhashed
    space or it evaluates just the input space, without applying the FFA.
    """

    import sys
    import MEN
    import utils

    # parameter input
    while True:
        spacefiles = utils.loop_input(rtype=str, default=None, msg="Space to be used (without file extension): ")
        try:
            data = spacefiles + ".dm"
            column_labels = spacefiles + ".cols"
            # returns {word:word_vector}
            unhashed_space = utils.readDM(data)
            # returns both-ways dicts of the vocabulary (word:index_in_vector)
            i_to_cols, cols_to_i = utils.readCols(column_labels)
        except FileNotFoundError as e:
            print("Unable to find files for input space and/or vocabulary.\n\
                   - correct file path?\n\
                   - are the file extensions '.dm' and '.cols'?\n\
                   - don't specify the file extension.")
            continue
        else:
            break
    MEN_annot = utils.loop_input(rtype=str, default=None, msg="Testset to be used: ")
    evaluate_mode = True if input("Only evaluate the space (without flying)? [y/n] ").upper() == "Y" else False
    if evaluate_mode is False:
        # Fruitfly parameters
        flattening = utils.loop_input(rtype=str, default="log",
                                      msg="Choose flattening function ([log, log2, log10] -- default: log): ")
        # pn_size is dictated by the size of the unhashed space
        pn_size = len(cols_to_i)
        print("Number of PNs:", pn_size)
        kc_size = utils.loop_input(rtype=int, default=10000, msg="Number of KCs (default: 10000): ")
        proj_size = utils.loop_input(rtype=int, default=6, msg="Number of projections per KC (default: 6): ")
        hash_percent = utils.loop_input(rtype=int, default=5, msg="Percentage of winners in the hash (default: 5): ")

        iterations = utils.loop_input(rtype=int, default=1, msg="How many runs (default: 1): ")
        verbose = True if input("Verbose mode (prints out important words)? [y/n] ").upper() == "Y" else False

    # executive code
    all_spb = []
    all_spa = []
    all_spd = []

    for i in range(iterations):
        if iterations > 1:
            print("\n#=== NEW RUN:", i + 1, "===#")
        # for pure evaluation of unhashed spaces
        if evaluate_mode:
            spb, count = MEN.compute_men_spearman(unhashed_space, MEN_annot)
            print("Performance:", round(spb, 4), "(calculated over", count, "items.)")
            sys.exit()

        # initiating and applying a Fruitfly object
        fruitfly = Fruitfly.from_scratch(pn_size, kc_size, proj_size, hash_percent, flattening)
        # this is where the magic happens
        space_hashed, space_dic, space_ind = fruitfly.fly(unhashed_space, cols_to_i)

        if verbose:
            for w in space_hashed:
                # prints out the words corresponding to a hashed word's most active PNs
                words = fruitfly.important_words_for(space_hashed[w], space_ind, n=6)
                print("{0} IMPORTANT WORDS: {1}".format(w, words))

        # evaluation and statistics
        spb, count = MEN.compute_men_spearman(unhashed_space, MEN_annot)
        print("Spearman before flying:", round(spb, 4), "(calculated over", count, "items.)")
        spa, count = MEN.compute_men_spearman(space_hashed, MEN_annot)
        print("Spearman after flying: ", round(spa, 4), "(calculated over", count, "items.)")
        print("difference:", round(spa - spb, 4))

        all_spb.append(spb)
        all_spa.append(spa)
        all_spd.append(spa - spb)

    if iterations > 1:
        best = sorted(all_spd, reverse=True)
        print("\nFinished all", iterations, "runs. Summary:")
        print("best and worst runs:", [round(e, 4) for e in best[:3]].extend([round(e, 4) for e in best[:-3]]))
        print("mean Sp. before:    ", round(np.average(all_spb), 4))
        print("mean Sp. after:     ", round(np.average(all_spa), 4))
        print("mean Sp. difference:", round(np.average(all_spd), 4))
        print("var of Sp. before:    ", round(float(np.var(all_spb, ddof=1), 8)))
        print("var of Sp. after:     ", round(float(np.var(all_spa, ddof=1), 8)))
        print("var of Sp. difference:", round(float(np.var(all_spd, ddof=1), 8)))
        print("std of Sp. before:     ", round(float(np.std(all_spb, ddof=1), 8)))
        print("std of Sp. after:      ", round(float(np.std(all_spa, ddof=1), 8)))
        print("std of Sp. difference: ", round(float(np.std(all_spd, ddof=1), 8)))

