"""Hyperopt. Perform a grid search over FFA parameter ranges for a given co-occurrence space.

Usage:
  hyperopt.py [--help] 
  hyperopt.py <space> <testset> 
  hyperopt.py <space> <testset> [--logto <log_dir>] [-v] [--no-summary]

Options:
  -h --help             show this screen
  -v --verbose          comment program status with command-line output
  --logto=<log_dir>     creates one file per run in the specified folder
  --no-summary          omit the creation of a final summary file
  --k1=<kc_min>          KC expansion factor, min< [default: 5]
  --k2=<kc_max>                               maximum [default: 5]
  --k3=<kc_step>                            step size [default: 1]
  --p1=<proj_min>        projection numbers per KC, minimum [default: 5]
  --p2=<proj_max>                                   maximum [default: 5]
  --p3=<proj_step>                                step size [default: 1]
  --r1=<hash_min>        percentages for hashing, minimum [default: 5]
  --r2=<hash_max>                                 maximum [default: 5]
  --r3=<hash_step>                              step size [default: 1]
  --flat=<flattenings>  flattening function(s), one of [log log2 log10], dash-separated [default: log]
"""

"""
This is a script for hyperparameter optimizaton of the Fruitfly Algorithm (FFA).
Given a space of a certain dimensionality, it performs a grid search over the
specified FFA parameter ranges. It produces 3 types of output files:
- Results, 1 per run
- Dump of results, incremental
- final summary, listing all results
"""

import os
import sys
import fcntl

import MEN
import utils
from Fruitfly import Fruitfly


"""
You can use docopt for easier usage if you run this script manually.
In order to use docopt, activate the following code and replace the 
'pass' statement with the rest of the script.

from docopt import docopt
if __name__ == '__main__':
    arguments = docopt(__doc__)
    pass
"""

if len(sys.argv) < 2:
    print("Check your parameters! Parameter sequence: \n\
        hyperopt.py \n\
        [space]               file path to vectors (embeddings/counts), without extension: \n\
        -testset [filepath]   default: data/MEN_dataset_natural_form_full\n\
        -logto [directory]    one file in [directory] per run; default: results/hyperopt/default_log\n\
        -kc [min max steps]   expansion factor; default: [5 5 1]\n\
        -proj [min max steps] number of projections; default: [5 5 1]\n\
        -hash [min max steps] percentage of 'winner' KCs; default: [5 5 1]\n\
        [flattenings]         any combination of [log log2 log10]; default: log\n\
        -no-summary           omit creation of a summary of all runs\n\
        -v                    run in verbose mode")
    sys.exit()

# ========== FUNCTIONS
def get_text_resources_from_argv():
    """
    :return: str -- file path to the co-occurrence count / space-to-be-hashed
    :return: str -- file path to the corresponding vocabulary file
    """
    data = sys.argv[1]+".dm"
    column_labels = sys.argv[1] + ".cols"
    return data, column_labels

def get_ranges_from_argv(param, minimum=5, maximum=5, steps=1):
    """
    Returns ranges for all FFA parameters except flattening:
    PN layer size, KC layer size, number of projections, and percentag of winners.
    Default values are [5, 5, 1] (= only the value 5)
    :param param: str -- key to the parameter range in argv
    :param minimum:
    :param maximum:
    :param steps:
    :return: minimum, maximum, steps
    """
    if param in sys.argv:
        minimum = int(sys.argv[sys.argv.index(param) + 1])
        maximum = int(sys.argv[sys.argv.index(param) + 2])
        steps = int(sys.argv[sys.argv.index(param) + 3])
    return minimum, maximum, steps

def get_flattening_from_argv():
    """
    Default value: log
    :return: [str] -- flattening functions (implemented: log, log2, log10)
    """
    flattening = []
    if "log" in sys.argv:
        flattening.append("log")
    if "log2" in sys.argv:
        flattening.append("log2")
    if "log10" in sys.argv:
        flattening.append("log10")
    if not flattening:
        flattening = ["log"]
    return flattening

def get_testset_from_argv():
    """
    Default value: "data/MEN_dataset_natural_form_full"
    :return: str -- file path to the test set (list of word pairs and associated similarity score)
    """
    if "-testset" in sys.argv:
        testfile = sys.argv[sys.argv.index("-testset") + 1]
    else:
        testfile = "data/MEN_dataset_natural_form_full"
    return testfile

def get_logging_from_argv():
    """
    Default value: "results/hyperopt/default_log"
    :return: str -- directory path to the folder containing all logs
    """
    if "-logto" in sys.argv:
        log_dest = sys.argv[sys.argv.index("-logto") + 1]
    else:
        log_dest = "results/hyperopt/default_log"
    if not os.path.isdir(log_dest):
        os.makedirs(log_dest, exist_ok=True)
    return log_dest

def evaluate(orig_space, result_space, goldstd):
    """
    Computes the Spearman correlation of both spaces, before and after "flying".
    Returns the number of word pairs taken for evaluation as well as the correlation values
    (before, after, improvement)
    :param orig_space: {str:[float]} -- words and their associated vectors (e.g. co-occurrence counts)
    :param result_space: {str:[int]} -- words and their associated binary hash signatures
    :param goldstd: str -- file path to the test set
    :return: int -- number of pairs in the unhashed space that were taken for evaluation
    :return: int -- number of pairs in the hashed space that were taken for evaluation
    :return: float -- spearman correlation between the unhashed space and the test set
    :return: float -- spearman correlation between the hashed space and the test set
    :return: float -- difference of the spearman correlation values (after-before)
    """
    sp_before, count_before = MEN.compute_men_spearman(orig_space, goldstd)
    sp_after, count_after = MEN.compute_men_spearman(result_space, goldstd)
    sp_diff = sp_after - sp_before
    return (count_before, count_after, sp_before, sp_after, sp_diff)

def log_results(results, ff_config, log_dest, result_space=None, pair_cos=True):
    """
    Creates one file to log the results of the current run (and optionally log
    similarities of word pairs in the evaluated space) and appends the core
    information to a dump file (in the case that the script doesn't finish, the
    dump contains the information in a readable format.)
    :param results: [int, int, float, float, float] -- set sizes and correlation values
    :param ff_config: {str:object} -- FFA parameter names and their values
    :param log_dest: str -- path to the log directory
    :param result_space: {str:[int]} -- words and their associated binary hash signatures
    :param pair_cos: bool -- optionally log cosine similarity values of evaluated word pairs
    """
    pns = ff_config["pn_size"]
    kcs = ff_config["kc_size"]
    proj = ff_config["proj_size"]
    hp = ff_config["hash_percent"]
    flat = ff_config["flattening"]

    logfilepath = log_dest + "/" + str(int(kcs / pns)) + "-" + str(proj) + "-" + \
                  str(int((hp * kcs) / 100)) + flat + "-" + ".txt"
    summarydump = log_dest + "/dump.txt"

    countb = results[0]
    counta = results[1]
    spb = round(results[2], 5)
    spa = round(results[3], 5)
    diff = round(results[4], 5)

    specs_statement = "\nPN_size \t" + str(pns) + \
                      "\nKC_factor\t" + str(kcs / pns) + \
                      "\nprojections\t" + str(proj) + \
                      "\nhash_dims\t" + str(hp * kcs / 100) + \
                      "\nflattening\t" + str(flat)
    results_statement = "testwords_before\t" + str(countb) + \
                        "testwords_after\t" + str(counta) + \
                        "\nsp_before\t" + str(spb) + \
                        "\nsp_after\t" + str(spa) + \
                        "\nsp_diff \t" + str(diff) + "\n"

    with open(logfilepath, "w") as f, open(summarydump, "a") as d:
        # for multiprocessing
        fcntl.flock(d, fcntl.LOCK_EX)
        f.write("Evaluated corpus:\t" + data + "\n")
        f.write(specs_statement + "\n" + results_statement + "\n")
        d.write(specs_statement + "\n" + results_statement + "\n")
        fcntl.flock(d, fcntl.LOCK_UN)

        # also log cosine similarities of the evaluated word pairs
        if (not (result_space is None) and (pair_cos is True)):
            pairs, men_sim, fly_sim = MEN.compile_similarity_lists(result_space, goldstandard)
            for i in range(len(pairs)):
                f.write(str(pairs[i][0]) + "\t" + str(pairs[i][1]) + "\t" + \
                        str(men_sim[i]) + "\t" + str(fly_sim[i]) + "\t" + "\n")
    if verbose:
        print(specs_statement)
        print(results_statement)

def log_final_results():
    """
    Logs a condensed version of the test results to a single file, sorted by performance.
    """
    with open(log_dest + "/summary.txt", "w") as f:
        # data structure of ranked_res: [(run,(results))]
        ranked_res = sorted(internal_log.items(), key=lambda x: x[1][3], reverse=True)
        summary_header = \
            "Grid search on the text data " + data + " with the following parameter ranges:\n" + \
            "KC factor (min, max, steps): {0} {1} {2}\n".format(kc_factor_min, kc_factor_max, kc_steps) + \
            "projections (min, max, steps): {0} {1} {2}\n".format(projections_min, projections_max, proj_steps) + \
            "hash percent (min, max, steps): {0} {1} {2}\n".format(hash_perc_min, hash_perc_max, hash_steps) + \
            "flattening functions: " + ", ".join(flattening) + "\n" + \
            "number of runs: " + str(len(ranked_res)) + "\n\n"
        f.write(summary_header)
        for run in ranked_res:
            f.write("{0}\t{1}\t{2}\tconfig: {3}".format(internal_log[run[0]][2],
                                                        internal_log[run[0]][3],
                                                        internal_log[run[0]][4],
                                                        all_ff_specs[run[0]]))
        if verbose:
            print("Best runs by performance:")
            for run in ranked_res[:min(10, int(round(len(ranked_res) / 10 + 1)))]:
                print("improvement:", round(internal_log[run[0]][4], 5), "with configuration:", all_ff_specs[run[0]])

""" Parameter Input """
data, column_labels = get_text_resources_from_argv()
goldstandard = get_testset_from_argv()
log_dest = get_logging_from_argv()

flattening = get_flattening_from_argv()
kc_factor_min, kc_factor_max, kc_steps = get_ranges_from_argv("-kc")
projections_min, projections_max, proj_steps = get_ranges_from_argv("-proj")
hash_perc_min, hash_perc_max, hash_steps = get_ranges_from_argv("-hash")

# returns {str:[float]}
in_space = utils.readDM(data)
# returns {int:str} and {str:i}
i_to_cols, cols_to_i = utils.readCols(column_labels)
# length of word vector (= input dimension)
pn_size = len(i_to_cols)

# for reporting purposes
verbose = "-v" in sys.argv
no_overall_summary_wanted = "-no-summary" in sys.argv
# {run:ff_specs}
all_ff_specs = {}
# {run:results}
internal_log = {}
sp_vals = {}

""" Grid Search"""
run = 0
for flat in flattening:
    for kc_factor in range(kc_factor_min, kc_factor_max + 1, kc_steps):
        for projections in range(projections_min, projections_max + 1, proj_steps):
            for hash_size in range(hash_perc_min, hash_perc_max + 1, hash_steps):
                run += 1
                # use fruitfly = Fruitfly.from_config(...) to work with the same FFA every time
                fruitfly = Fruitfly.from_scratch(pn_size, kc_factor * pn_size, projections, hash_size, flat)
                if verbose: print("Run number {0}; config: {1}".format(run, fruitfly.show_off()))
                # this is where the magic happens
                out_space, space_dic, space_ind = fruitfly.fly(in_space, cols_to_i)
                internal_log[run] = evaluate(in_space, out_space, goldstandard)
                # log externally and keep track in the summary
                log_results(internal_log[run], fruitfly.get_specs(), log_dest, out_space)
                all_ff_specs[run] = fruitfly.get_specs()

if verbose: print("Finished grid search. Number of runs:", run)
if no_overall_summary_wanted is False:
    log_final_results()
print("done.")
