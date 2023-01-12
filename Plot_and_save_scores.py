import os
import sys
from sklearn.cluster import KMeans
import networkx as nx
import numpy as np
from graph_generation.load_graphs_and_create_metadata import dataset_metadata
from graph_matching_tools.metrics import matching
import matplotlib.pyplot as plt
import scipy.io as sio


def score_mean_std(scores):
    
    avg_scores = []
    std_scores = []

    for keys,values in scores.items():
        avg_scores.append(np.mean(values))
        std_scores.append(np.std(values))
        
    return np.array(avg_scores), np.array(std_scores)


def plot_accuracy():




if __name__ == '__main__':

	# Load all F1 Scores

	kergm_scores = nx.read_gpickle('NEW_SIMU_KerGM_score.gpickle')
	m_Sync_scores = nx.read_gpickle('NEW_SIMU_mSync_score.gpickle')
	mALS_scores = nx.read_gpickle('NEW_SIMU_mALS_score.gpickle')
	CAO_scores = nx.read_gpickle('NEW_SIMU_CAO_score.gpickle')

	# Load all Precision Scores

	kergm_scores_prec = nx.read_gpickle('NEW_SIMU_KerGM_score_prec_scores.gpickle')
	m_Sync_scores_prec = nx.read_gpickle('NEW_SIMU_mSync_score_prec_scores.gpickle')
	mALS_score_prec = nx.read_gpickle('NEW_SIMU_mALS_score_prec_scores.gpickle')
	CAO_score_prec = nx.read_gpickle('NEW_SIMU_CAO_score_prec_scores.gpickle')

	# Load all Recall Scores

	kergm_scores_rec = nx.read_gpickle('NEW_SIMU_KerGM_score_rec_scores.gpickle')
	m_Sync_scores_rec = nx.read_gpickle('NEW_SIMU_mSync_score_rec_scores.gpickle')
	mALS_score_rec = nx.read_gpickle('NEW_SIMU_mALS_score_rec_scores.gpickle')
	CAO_score_rec = nx.read_gpickle('NEW_SIMU_CAO_score_rec_scores.gpickle')

	# Compute the mean and std for all F1 Scores
	kergm_mean, kergm_std = score_mean_std(kergm_scores)
	msync_mean, msync_std = score_mean_std(m_Sync_scores)
	mALS_mean, mALS_std = score_mean_std(mALS_scores)
	CAO_mean, CAO_std = score_mean_std(CAO_scores)

	# Compute the mean and std for all Precision Scores
	kergm_prec, kergm_prec_std = score_mean_std(kergm_scores_prec)
	msync_prec, msync_prec_std = score_mean_std(m_Sync_scores_prec)
	mALS_prec, mALS_prec_std = score_mean_std(mALS_score_prec)
	CAO_prec, CAO_prec_std = score_mean_std(CAO_score_prec)

	# Compute the mean and std for all Precision Scores
	kergm_rec, kergm_rec_std = score_mean_std(kergm_scores_rec)
	msync_rec, msync_rec_std = score_mean_std(m_Sync_scores_rec)
	mALS_rec, mALS_rec_std = score_mean_std(mALS_score_rec)
	CAO_rec, CAO_rec_std = score_mean_std(CAO_score_rec)


	# Plot Figures ------------------------------------------------------------------------------------------

	strkergm = [str(k) for k in list(kergm_scores.keys())]



	# Plot all F1 Scores

	plt.plot(strkergm, kergm_mean ,label = 'KerGM')
	plt.fill_between(strkergm, kergm_mean - kergm_std, kergm_mean + kergm_std, alpha=0.2)

	plt.plot(strkergm, msync_mean ,label = 'mSync')
	plt.fill_between(strkergm, msync_mean - msync_std, msync_mean + msync_std, alpha=0.2)

	plt.plot(strkergm, mALS_mean ,label = 'mALS')
	plt.fill_between(strkergm, mALS_mean - mALS_std, mALS_mean + mALS_std, alpha=0.2)

	plt.plot(strkergm, CAO_mean ,label = 'CAO')
	plt.fill_between(strkergm, CAO_mean - CAO_std, CAO_mean + CAO_std, alpha=0.2)

	plt.xlabel('kappa',fontweight="bold",fontsize=28)
	plt.ylabel('F1 score',fontweight="bold",fontsize=28)
	plt.legend(loc = 'lower left')
	#plt.title('F1 Score curve',fontweight="bold",fontsize=24)
	plt.gca().yaxis.grid(True)
	plt.gca().invert_xaxis()
	plt.legend(loc=3, prop={'size': 18})
	plt.xticks(fontsize=20)
	plt.yticks(np.arange(0,1.1,0.1),fontsize=20)

	plt.show()
	fig.savefig('F1_Scores.png')



	# Plot all Precision Scores

	plt.plot(strkergm, kergm_prec ,label = 'KerGM')
	plt.fill_between(strkergm, kergm_prec - kergm_prec_std, kergm_prec + kergm_prec_std, alpha=0.2)

	plt.plot(strkergm, msync_prec ,label = 'mSync')
	plt.fill_between(strkergm, msync_prec - msync_prec_std, msync_prec + msync_prec_std, alpha=0.2)

	plt.plot(strkergm, mALS_prec ,label = 'mALS')
	plt.fill_between(strkergm, mALS_prec - mALS_prec_std, mALS_prec + mALS_prec_std, alpha=0.2)

	plt.plot(strkergm, CAO_prec ,label = 'CAO')
	plt.fill_between(strkergm, CAO_prec - CAO_prec_std, CAO_prec + CAO_prec_std, alpha=0.2)


	plt.xlabel('kappa',fontweight="bold",fontsize=28)
	plt.ylabel('Precision',fontweight="bold",fontsize=28)
	plt.legend(loc = 'lower left')
	plt.gca().yaxis.grid(True)
	plt.gca().invert_xaxis()
	plt.legend(loc=3, prop={'size': 18})
	plt.xticks(fontsize=20)
	plt.yticks(np.arange(0,1.1,0.1),fontsize=20)

	plt.show()
	fig.savefig('Precision_scores.png')



	# Plot all Recall Scores

	plt.plot(strkergm, kergm_rec ,label = 'KerGM')
	plt.fill_between(strkergm, kergm_rec - kergm_rec_std, kergm_rec + kergm_rec_std, alpha=0.2)

	plt.plot(strkergm, msync_rec ,label = 'mSync')
	plt.fill_between(strkergm, msync_rec - msync_rec_std, msync_rec + msync_rec_std, alpha=0.2)

	plt.plot(strkergm, mALS_rec ,label = 'mALS')
	plt.fill_between(strkergm, mALS_rec - mALS_rec_std, mALS_rec + mALS_rec_std, alpha=0.2)

	plt.plot(strkergm, CAO_rec ,label = 'CAO')
	plt.fill_between(strkergm, CAO_rec - CAO_rec_std, CAO_rec + CAO_rec_std, alpha=0.2)

	plt.xlabel('kappa',fontweight="bold",fontsize=28)
	plt.ylabel('Recall',fontweight="bold",fontsize=28)
	plt.legend(loc = 'lower left')
	plt.gca().yaxis.grid(True)
	plt.gca().invert_xaxis()
	plt.legend(loc=3, prop={'size': 18})
	plt.xticks(fontsize=20)
	plt.yticks(np.arange(0,1.1,0.1),fontsize=20)

	plt.show()
	fig.savefig('Recall_scores.png')
