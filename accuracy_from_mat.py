import os
import sys
from sklearn.cluster import KMeans
import networkx as nx
import numpy as np
from graph_generation.load_graphs_and_create_metadata import dataset_metadata
from graph_matching_tools.metrics import matching
import matplotlib.pyplot as plt
import scipy.io as sio

def get_permutation_matrix_from_dictionary(matching, g_sizes):
    """
    Create the full permutation matrix from the matching result
    :param matching: the matching result for each graph (nodes number, assignment)
    :param g_sizes: the list of the size of the different graph
    :return: the full permutation matrix
    """
    f_size = int(np.sum(g_sizes))
    res = np.zeros((f_size, f_size))

    idx1 = 0
    for i_g1 in range(len(g_sizes)):
        idx2 = 0
        for i_g2 in range(len(g_sizes)):
            match = matching["{},{}".format(i_g1, i_g2)]
            for k in match:
                res[idx1 + int(k), idx2 + match[k]] = 1
            idx2 += g_sizes[i_g2]
        idx1 += g_sizes[i_g1]
        
    np.fill_diagonal(res,1)
    return res


def score_mean_std(scores):
    
    avg_scores = []
    std_scores = []

    for keys,values in scores.items():
        avg_scores.append(np.mean(values))
        std_scores.append(np.std(values))
        
    return np.array(avg_scores), np.array(std_scores)


def accu_from_mat(path_to_graph_folder, path_to_dummy_graph_folder, mat_file):

	scores = {100:[],200:[],400:[],1000:[]}
	prec_scores = {100:[],200:[],400:[],1000:[]}
	rec_scores = {100:[],200:[],400:[],1000:[]}


	for trial in trials:
		print('trial: ', trial)
    
    	all_files = os.listdir(path_to_graph_folder+trial)
    
    	for folder in all_files:
        
        
        	if os.path.isdir(path_to_graph_folder+trial+'/'+ folder):
            
            	print('Noise folder: ',folder)
            
            	path_to_graphs = path_to_graph_folder + '/' + trial + '/' + folder+'/graphs/'
            	path_to_dummy_graphs = path_to_dummy_graph_folder + '/' + trial +'/' + folder + '/0/graphs/'
            	path_to_groundtruth_ref = path_to_graph_folder + '/' + trial +'/' + folder + '/permutation_to_ref_graph.gpickle'
            	path_to_groundtruth  = path_to_graph_folder + '/' + trial + '/' + folder + '/ground_truth.gpickle'
            
            	noise = folder.split(',')[0].split('_')[1]
            

            	graph_meta = dataset_metadata(path_to_graphs, path_to_groundtruth_ref)
            	ground_truth =  nx.read_gpickle(path_to_groundtruth)   
            	res = get_permutation_matrix_from_dictionary(ground_truth, graph_meta.sizes)
            
            
            	all_dummy_graphs = [nx.read_gpickle(path_to_dummy_graphs+'/'+g) for g in np.sort(os.listdir(path_to_dummy_graphs))]
            	X_mat = sio.loadmat(path_to_graph_folder + '/' + trial + '/' + folder +'/'+ mat_file)['X']            
            	dummy_mask = [list(nx.get_node_attributes(graph,'is_dummy').values()) for graph in all_dummy_graphs]
            	dummy_mask = sum(dummy_mask,[])
            	dummy_indexes = [i for i in range(len(dummy_mask)) if dummy_mask[i]==True]            
            	X_mat = np.delete(X_mat,dummy_indexes,0) # delete the dummy rows
            	X_mat = np.delete(X_mat,dummy_indexes,1) # delete the dummy columns
            
            
            	print('res shape: ',res.shape)
            	print('X shape: ',X_mat.shape)
               
            
            	f1, prec, rec = matching.compute_f1score(X_mat,res)
            
            	scores[int(noise)].append(f1)
            	prec_scores[int(noise)].append(prec)
            	rec_scores[int(noise)].append(rec)

    return scores, prec_scores, rec_scores





if __name__ == '__main__':


	path_to_graph_folder = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/simu_graph/NEW_SIMUS_JULY_11/'
	path_to_dummy_graph_folder = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/simu_graph/NEW_SIMUS_JULY_11_with_dummy/'

	mat_file = 'X_cao_cst_o.mat'


	scores,prec_scores,rec_scores = accu_from_mat(path_to_graph_folder, path_to_dummy_graph_folder, mat_file)

	F1_file = 'F1_score_' + mat_file.split('.')[0]


	nx.write_gpickle(scores,'NEW_SIMU_CAO_score'+'.gpickle')
	nx.write_gpickle(prec_scores,'NEW_SIMU_CAO_score_prec_scores'+'.gpickle')
	nx.write_gpickle(rec_scores,'NEW_SIMU_CAO_score_rec_scores'+'.gpickle')

