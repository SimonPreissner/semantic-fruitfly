import os
import sys
import time

import math
import numpy as np
from tqdm import tqdm

import MEN
import utils
from Fruitfly import Fruitfly
from Incrementor import Incrementor

"""parameters (via console input)"""
# paths
global corpus_dir, testset_file, overlap_file,  \
       pipedir, \
       count_file, fly_dir, space_dir, results_dir, vip_words_dir, \
       w2v_corpus_file, w2v_exe_file, w2v_space_dir, w2v_results_file, \
       summary_file
# Fruitfly parameters
global pns, kcs, con, red, flat, max_pns
# Incrementor parameters
global window, max_dims, tokenize, postag_simple, min_count, \
       test_interval, vip_words_n, verbose
# variables/objects
global run, breeder, t_thisrun, performance_summary

#========== Initial parameter input and setup

def param_input_files():
    global corpus_dir, testset_file, overlap_file,  \
           pipedir, \
           count_file, fly_dir, space_dir, vip_words_dir, results_dir, \
           w2v_corpus_file, w2v_exe_file, w2v_space_dir, w2v_results_file, \
           summary_file
    print("=== Resources and Output Directories ===")
    corpus_dir = utils.loop_input(rtype=str, default="data/chunks_wiki",
                                  msg="Path to text resources (default: data/chunks_wiki): ")
    w2v_exe_file = utils.loop_input(rtype=str, default=None,
                                    msg="Path to Word2Vec code (optional): ")
    testset_file = utils.loop_input(rtype=str, default="data/MEN_dataset_natural_form_full",
                                    msg="Path to test set (default: data/MEN_dataset_natural_form_full): ")
    overlap_file = utils.loop_input(rtype=str, default=None,
                                    msg="Path to a word list to be checked for overlap (optional): ")
    pipedir = utils.loop_input(rtype=str, default="results/pipe00",
                               msg="Path to the resulting directories (default: results/pipe_00): ")
    summary_file = pipedir + "/" + "summary.tsv"
    results_dir = pipedir + "/" + "ffa/results/stats/"
    w2v_results_file = pipedir + "/" + "w2v/results.txt" if w2v_exe_file is not None else None
    count_file = pipedir + "/" + "ffa/cooc"
    fly_dir = pipedir + "/" + "ffa/configs/"
    space_dir = pipedir + "/" + "ffa/spaces/"
    vip_words_dir = pipedir + "/" + "ffa/results/words/"
    w2v_corpus_file = pipedir + "/" + "w2v/resource.txt" if w2v_exe_file is not None else None
    w2v_space_dir = pipedir + "/" + "w2v/" if w2v_exe_file is not None else None
    for f in [fly_dir, space_dir, results_dir, vip_words_dir, w2v_space_dir]:
        if f is not None and not os.path.isdir(f):
            os.makedirs(f, exist_ok=True)

def param_input_FFA():
    global pns, kcs, con, red, flat, max_pns
    print("=== Fruitfly Parameters ===")
    # Initial Fruitfly parameters (taken from the ukwac_100m gridsearch on 10k dims)
    pns, kcs, con, red, flat, max_pns = 50, 40000, 20, 8, "log", 10000
    d = True if input("use default FFA parameters ({0}, {1}, {2}, {3}, {4}, {5}) [y/n]? "
                      .format(pns, kcs, con, red, flat, max_pns)).upper() == "Y" else False
    if d is False:
        pns = utils.loop_input(rtype=int, default=pns, msg="Initial number of PNs: ")
        kcs = utils.loop_input(rtype=int, default=kcs, msg="Number of KCs: ")
        con = utils.loop_input(rtype=int, default=con, msg="Connections per KC: ")
        red = utils.loop_input(rtype=int, default=red, msg="WTA-percentage (whole number): ")
        flat = utils.loop_input(rtype=str, default=flat, msg="Flattening function (log/log2/log10): ")
        max_pns = utils.loop_input(rtype=int, default=max_pns, msg="Maximum number of PNs: ")

def param_input_misc():
    global window, max_dims, tokenize, postag_simple, min_count, \
           test_interval, vip_words_n, verbose
    print("=== Other Parameters ===")
    window = utils.loop_input(rtype=int, default=5, msg="Window size (to each side) for counting (default: 5): ")
    min_count = utils.loop_input(rtype=int, default=None,
                msg="Periodic deletion of words with n occurrences or fewer from the count (optional) -- n: ")
    max_dims = utils.loop_input(rtype=int, default = None,
                                msg="Maximum vocabulary size for the count (skip this for true incrementality): ")
    test_interval = utils.loop_input(rtype=int, default=None,
                                     msg="Test interval in words; leave empty for filewise testing): ")
    vip_words_n = utils.loop_input(rtype=int, default=50,
                                   msg="Number of important words to be extracted (default: 50): ")
    tokenize = False if input("Tokenize the input text? (default: yes) [y/n]").upper() == "N" else True
    postag_simple = True if input("Only count NN/VB/ADJ (requires pos-tagged testset for evaluation)? [y/n]").upper() == "Y" else False
    verbose = False if input("Be verbose while running? (default: yes) [y/n]").upper() == "N" else True

def setup_loop_environment():
    """
    Compiles a list of file paths of the resource files.
    Instantiates and logs an initial Fruitfly object.
    Instantiates the Incrementor object (without text resources)
    :return: [str] -- list of file paths
    :return: Incrementor -- main component of the script
    """
    # Make a list of file paths to be passed one by one to an Incrementor object
    #try:
    corpus_files = []
    if os.path.isfile(corpus_dir):
        corpus_files = [corpus_dir]
    else:
        for (dirpath, dirnames, filenames) in os.walk(corpus_dir):
            corpus_files.extend([dirpath + "/" + f for f in filenames])
    # Create the initial FFA beforehand because Incrementor can't do custom FFA initialization
    initial_fly_file = fly_dir + "fly_run_0.cfg"
    first_fly = Fruitfly.from_scratch(pn_size=pns, kc_size=kcs, proj_size=con,
                                      hash_percent=red, max_pn_size=max_pns, flattening=flat)
    first_fly.log_params(filename=initial_fly_file, timestamp=False)
    """
    The constructor of Incrementor 
    1) reads in text files from a directory (if specified)
    2) computes a frequency distribution over the current text 
    3) sets up or loads a Fruitfly object from an already-logged Fruitfly
    4) sets up or loads a co-occurence matrix, with a limit if specified
    - matrix_incremental=False because it would need to load an existing matrix; 
      however, in this setup, as only one Incrementor object is used, 
      it can increment all the way from an initial matrix.
    - corpus_linewise=False because it cannot be handled by the loop as of now. 
    - Initialization without corpus. File paths are passed to the Incrementor via the loop.
    """
    breeder = Incrementor(None, count_file,
                          corpus_tokenize=tokenize, corpus_linewise=False, corpus_checkvoc=overlap_file,
                          matrix_incremental=False, matrix_maxdims=max_dims, min_count=min_count,
                          contentwords_only=postag_simple, fly_new=False, fly_grow=True, fly_file=initial_fly_file,
                          verbose=verbose)
    #except Exception as e:
    #    with open(errorlog, "a") as f:
    #        f.write(str(e)[:500]+"\n")
    #    print("An error occured while setting up the loop environment. Check", errorlog, "for further information.")
    #    sys.exit()

    return corpus_files, breeder

#========== Core functionalities within the loop

def new_paths(run):
    """
    Adds the current run's number to file paths to make the run-specific file paths
    for the fruitfly, the hashed space, the results, and the important words.
    Returns these 4 file paths as strings.
    :param run: int -- the current run of the loop
    """
    a = fly_dir + "fly_run_" + str(run) + ".cfg"  # for logging
    b = space_dir + "space_run_" + str(run) + ".dh" # the hashes
    c = results_dir + "stats_run_" + str(run) + ".txt"
    d = vip_words_dir + "words_run_" + str(run) + ".txt"
    return a,b,c,d

def count_test_log(count_these):
    """
    The core part of the loop setting.
    Lets an Incrementor object count co-occurrences (and maintain a Fruitfly object).
    Saves the new count to a file and applies the FFA to it.
    Evaluates the resulting hashes and logs them along with the FFA.
    If specified, trains a Word2Vec model on the accumulated text data.
    :param count_these: [str] -- tokens to be taken into the co-occurrence count
    """
    # only counts co-occurrences of words within breeder.freq (which can be limited by matrix_maxdims)
    t_count = breeder.count_cooccurrences(words=count_these, window=window, timed=True)
    # delete very infrequent words from the count matrix
    nr_of_del_dims, t_cooc_del = breeder.reduce_count_size(verbose=verbose, timed=True)
    breeder.log_matrix()

    fly_these, unhashed_space, words_to_i = prepare_flight()
    # space_dic and space_ind are the words_to_i and i_to_words of the cropped vectors (done in Fruitfly.fit_space())
    hashed_space, space_dic, space_ind, t_flight = breeder.fruitfly.fly(fly_these, words_to_i, timed=True)
    eval_and_log_FFA(count_these, hashed_space, space_ind, t_count, t_cooc_del, t_flight, unhashed_space)

    if w2v_exe_file is not None:
        w2v_min_count, w2v_space_file, w2v_vocab_file = prepare_w2v(count_these)
        t_train = execute_w2v(w2v_min_count, w2v_space_file, w2v_vocab_file)
        eval_and_log_w2v(t_train, w2v_space_file)

#========== FFA application functions

def prepare_flight():
    """
    read in the count vectors etc. and choose which ones to fly
    based on a list of required words.
    :return fly_these: {str:[float]} -- possibly a subset of unhashed_space
    :return unhashed_space: {str:[float]} -- words and their corresponding co-occurrence counts
    :return_words_to_i: {str:int} -- mapping of context words to their position in the count
    """
    unhashed_space = utils.readDM(breeder.outspace)
    i_to_words, words_to_i = utils.readCols(breeder.outcols)
    # only select words that will be needed for evaluation:
    if overlap_file is None:
        # in this case, fly() is applied to the whole of unhashed_space
        fly_these = unhashed_space
    else:
        words_for_flight = breeder.read_checklist(overlap_file)
        fly_these = {w: unhashed_space[w] for w in words_for_flight if w in unhashed_space}
    return fly_these, unhashed_space, words_to_i

def eval_and_log_FFA(count_these, hashed_space, space_ind, t_count, t_cooc_del, t_flight, unhashed_space):
    """
    Evaluates hashes that were produced by a Fruitfly object. The evaluation metric is
    the Spearman correlation. Evaluation is designed to use the MEN dataset described in
    Bruni et al. (2014).
    Logs the results along with times taken for counting and "flying", and other stats.
    :param count_these: [str] -- tokens to be taken into the co-occurrence count
    :param hashed_space: {str:[int]} -- words and their associated hash signatures
    :param space_ind: {int:str} -- mapping of index:word for the input to the FFA
    :param t_count: float -- time taken to count co-occurrences
    :param t_cooc_del: float -- time taken to delete infrequent words from the count
    :param t_flight: float -- time taken to apply the FFA
    :param unhashed_space: {str:[float]} -- words and their corresponding co-occurrence counts
    """
    global performance_summary

    # Evaluate the hashed space
    if verbose: print("Evaluating against",testset_file,"...")
    spb, tsb = MEN.compute_men_spearman(unhashed_space, testset_file)
    spa, tsa = MEN.compute_men_spearman(hashed_space, testset_file)
    sp_diff = spa - spb
    if verbose: print("Spearman correlations of the spaces (before/after flying, diff):", spb,"/", spa, sp_diff)
    connectednesses = [len(cons) for cons in breeder.fruitfly.pn_to_kc.values()]
    avg_PN_con = round(sum(connectednesses) / breeder.fruitfly.pn_size, 6)
    var_PN_con = round(float(np.var(connectednesses, ddof=1)), 6)
    std_PN_con = round(float(np.std(connectednesses, ddof=1)), 6)

    t_thisrun = time.time() - t0_thisrun

    # Log the hashed space
    utils.writeDH(hashed_space, space_file)  # space_file is updated above
    # Log words of the most important PNs for each word
    if verbose: print("Logging most important words to", vip_words_file, "...")
    with open(vip_words_file, "w") as f:
        for w in tqdm(hashed_space):
            vip_words = breeder.fruitfly.important_words_for(hashed_space[w], space_ind, n=vip_words_n)
            vip_words_string = ", ".join(vip_words)
            f.write("{0} --> {1}\n".format(w, vip_words_string))
    # logs parameters and connections to a new file (breeder.flyfile is updated at the very beginning of the run)
    breeder.log_fly()
    # log the results
    with open(results_file, "w") as f:
        result_string = "RESULTS FOR RUN " + str(run) + ":\n" + \
                        "\nsp_corr_diff:       " + str(sp_diff) + \
                        "\nsp_corr_after:      " + str(spa) + \
                        "\nsp_corr_before:     " + str(spb) + \
                        "\nitems_tested_after: " + str(tsa) + \
                        "\nitems_tested_before:" + str(tsb) + "\n"
        # fly config
        fc = breeder.fruitfly.get_specs()
        t = [fc["pn_size"], fc["kc_size"], fc["proj_size"], fc["hash_percent"], fc["flattening"], fc["max_pn_size"]]
        fruitfly_string = "\nFFA_CONFIG: (" + ", ".join([str(i) for i in t]) + ")" + \
                          "\navg_PN_connections: " + str(avg_PN_con) + \
                          "\nvar_PN_connections: " + str(var_PN_con) + \
                          "\nstd_PN_connections: " + str(std_PN_con) + "\n"
        time_string = "\nTIME TAKEN: " + \
                      "\ncounting: " + str(t_count) + \
                      "\nreducing: " + str(t_cooc_del) + \
                      "\nflying:   " + str(t_flight) + \
                      "\ntotal:    " + str(t_thisrun) + "\n"
        file_string = "\nRELATED FILES:" \
                      "\nfruitfly:        " + breeder.flyfile + \
                      "\nhashed_space:    " + space_file + \
                      "\nimportant_words: " + vip_words_file + "\n"
        f.write(result_string + fruitfly_string + time_string + file_string)
    # keep internal log
    performance_summary[run] = [len(count_these), fc["pn_size"], tsb, spb, spa, sp_diff, t_thisrun]

#========== Word2Vec application functions

def prepare_w2v(count_these):
    """
    Prepares the input file, the minimum count, and the output file names for
    the execution of Word2Vec code.
    :param count_these: [str] -- list of tokens to be added to a .txt file.
    :return w2v_min_count: int -- minimum occurrences of a word in the text data to be taken into account by w2v
    :return w2v_space_file: str -- file path of the w2v model to be trained
    :return w2v_vocab_file: str -- file path to the vocabulary of the w2v embeddings
    """
    if verbose: print("\nPreparing to run Word2Vec ...\n")
    # add the current text slice to a file that can be used by w2v
    with open(w2v_corpus_file, "a") as f:
        f.write(" ".join(count_these) + " ")
    # choose the minimum word count for the w2v run: it's connected to the lowest count in the FFA's PN layer
    occs = sorted([sum(vec) for vec in breeder.cooc])[:breeder.fruitfly.max_pn_size]
    # selects the smallest number
    w2v_min_count = math.floor(occs[0] / (window * 2))
    w2v_space_file = w2v_space_dir + "space.txt"
    w2v_vocab_file = w2v_space_dir + "space.vocab"
    return w2v_min_count, w2v_space_file, w2v_vocab_file

def execute_w2v(w2v_min_count, w2v_space_file, w2v_vocab_file):
    """
    Runs Word2Vec code with a terminal command.
    :param w2v_min_count: int -- minimum occurrences of a word in the text data to be taken into account by w2v
    :param w2v_space_file: str -- file path of the w2v model to be trained
    :param w2v_vocab_file: str -- file path to the vocabulary of the w2v embeddings
    :return: float -- time taken to execute the Word2Vec code
    """
    if verbose: print("training Word2Vec with minimum count", w2v_min_count, "...")
    # run the w2v code
    t0 = time.time()
    try:
        os.system(w2v_exe_file + " " +
                  "-train " + w2v_corpus_file +
                  "-output " + w2v_space_file +
                  "-size 300 " +
                  "-window " + str(window) +
                  "-sample 1e-3 " +
                  "-negative 10 " +
                  "-iter 1 " +
                  "-min-count " + str(w2v_min_count) +
                  "-save-vocab " + w2v_vocab_file)
    except Exception as e:
        with open(errorlog, "a") as f:
            f.write(str(e)[:500]+"\n")
        print("An error occured while running Word2Vec. Check", errorlog, "for further information.")
        print("Continuing without running Word2Vec ...")
    return time.time() - t0

def eval_and_log_w2v(t_train, w2v_space_file):
    """
    Evaluates the trained Word2Vec model. The evaluation metric is the Spearman correlation.
    Evaluation is designed to use the MEN dataset described in Bruni et al. (2014).
    :param t_train: float -- time taken to train the Word2Vec model
    :param w2v_space_file: str -- file path of the w2v model that was trained
    """
    global performance_summary
    if verbose: print("evaluating and logging the Word2Vec model ...")
    # evaluate the w2v model
    try:
        w2v_space = utils.readDM(w2v_space_file)
        spcorr, pairs = MEN.compute_men_spearman(w2v_space, testset_file)
        with open(w2v_results_file, "a+") as f:
            f.write("RUN " + str(run) + ":" + \
                    "\tSP_CORR: " + str(spcorr) + \
                    "\tTEST_PAIRS: " + str(pairs) + \
                    "\tTRAIN_TIME: " + str(t_train) + "\n")
        # keep internal log
        performance_summary[run].extend([spcorr, pairs, t_train])
    except Exception as e:
        with open(errorlog, "a") as f:
            f.write(str(e)[:500]+"\n")
        print("An error occured while evaluating Word2Vec. Check", errorlog, "for further information.")


#==========

def log_final_summary(run_to_stats):
    """
    Creates a file that contains the most important statistics of every run.
    This is useful for an overview of a setup with many iterations (=runs)
    :param run_to_stats: {int:[float]} -- sore statistics collected throughout the runs
    """
    if verbose: print("\n\nFinished all runs.")
    column_labels = ["run",
                     "new_data_in_words", "PN_size", "testset",
                     "sp.corr_before", "sp.corr_after", "sp_diff",
                     "time-taken", "w2v_score", "w2v_testset"]
    # +1 because the first row is the run (which is not contained in the values of the summary dict)
    row_length = len(run_to_stats[list(run_to_stats.keys())[0]])+1

    with open(summary_file, "w") as f:
        # summary header
        f.write("\t".join(column_labels[:row_length]) + "\n\n")
        # summary body
        for k, stat_tuple in run_to_stats.items():
            f.write(str(k) + "\t" + "\t".join([str(v) for v in stat_tuple]) + "\n")



#========== END OF FUNCTIONS

if __name__ == '__main__':
    """
    This is the full pipeline of the incremental Fruitfly.
    It reads text resources as needed,
    counts co-occurrences and grows a Fruitfly alongside,
    then applies the FFA to the counts,
    optionally runs Word2Vec on the same corpus,
    evaluates the results, and logs them.
    """

    errorlog = "spacebreeder_errorlog.txt"

    # parameter input via terminal
    param_input_files()
    param_input_FFA()
    param_input_misc()
    print("=== Let's go! ===")
    runtime_zero = time.time()

    # set up resource paths, initial Fruitfly, and Incrementor
    corpus_files, breeder = setup_loop_environment()
    # for a final summary
    performance_summary = {}
    # wc = word count; used for slicing breeder.words
    t0_thisrun, wc, run = 0, 0, 0

    for (file_n, file) in enumerate(corpus_files):
        # read in a new file (but no counting yet)
        breeder.extend_corpus(file)

        # Count and test in word intervals
        if test_interval is not None:
            # Last run with the remaining words of breeder.words
            if file_n+1 == len(corpus_files) and len(breeder.words) < wc+test_interval:
                # ends before logging
                t0_thisrun = time.time()
                run += 1
                if verbose: print("\n\nStarting the last run ...")
                breeder.flyfile, space_file, results_file, vip_words_file = new_paths(run)
                # take the whole rest of breeder.words()
                count_these = breeder.words[wc:len(breeder.words)]
                if verbose: print("Size of the final corpus slice:", len(count_these))
                # this is the actual loop content
                count_test_log(count_these)
                if verbose: print("End of the last run.\n\n")

            # Loop as long as there are enough words for a whole slice
            while len(breeder.words) >= wc+test_interval:
                # ends before logging
                t0_thisrun = time.time()
                run = int(wc / test_interval) + 1
                if verbose: print("\n\nStarting run",run,"...")
                breeder.flyfile, space_file, results_file, vip_words_file = new_paths(run)
                count_these = breeder.words[wc: wc + test_interval]
                if verbose: print("Size of corpus slice:", len(count_these))
                # this is the actual loop content
                count_test_log(count_these)
                wc += test_interval
                if verbose: print("End of run "+str(run)+".\n\n")

        # Count and test once per file
        else:
            # ends before logging
            t0_thisrun = time.time()
            run += 1
            if verbose: print("\n\nStarting run", run, "with file", file, "...")
            breeder.flyfile, space_file, results_file, vip_words_file = new_paths(run)
            # count the whole newly read-in document
            count_these = breeder.words[wc:len(breeder.words)]
            if verbose: print("Size of text resource in this run:", len(count_these))
            # this is the actual loop content
            count_test_log(count_these)
            wc = len(breeder.words)
            if verbose: print("End of run " + str(run) + ".\n\n")

    log_final_summary(performance_summary)
    print("Finished all runs. Total time taken:",time.time()-runtime_zero,"\ndone.")




