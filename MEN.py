#Evaluate semantic space against MEN dataset

from scipy import stats

import utils


#Note: this is scipy's spearman, without tie adjustment
def spearman(x,y):
	return stats.spearmanr(x, y)[0]

def readMEN(annotation_file):
	pairs=[]
	humans=[]
	with open(annotation_file,'r') as f:
		for l in f:
			l=l.rstrip('\n')
			items=l.split()
			pairs.append((items[0],items[1]))
			humans.append(float(items[2]))
	return pairs, humans

def compile_similarity_lists(dm_dict, annotation_file):
	pairs, humans=readMEN(annotation_file)
	system_actual=[]
	human_actual=[]
	eval_pairs=[]

	for i in range(len(pairs)):
		human=humans[i]
		a,b=pairs[i]
		if a in dm_dict and b in dm_dict:
			cos=utils.cosine_similarity(dm_dict[a],dm_dict[b])
			system_actual.append(cos)
			human_actual.append(human)
			eval_pairs.append(pairs[i])

	return eval_pairs, human_actual, system_actual

def compute_men_spearman(dm_dict, annotation_file):
	eval_pairs, human_actual, system_actual = compile_similarity_lists(dm_dict, annotation_file)
	count = len(eval_pairs)
	sp = spearman(human_actual,system_actual)
	return sp,count

