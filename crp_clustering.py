import random
import numpy
import scipy.stats as stats
import scipy.cluster.vq as vq

def sample_with_weights(weights, sum_weights=None):
	"""Sample from a multinomial distribution

	Args:
		weights - a numpy array of positive numbers of associated weights for each index
		sum_weights - the sum of the above list. if we have call this function many times
			on the same weight, providing the sum will save a lot of computation time

	Returns:
		the index that gets chosen.
		-1 if a weight is invalid
	"""
	if sum_weights is None:
		sum_weights = numpy.sum(weights)
	p = random.uniform(0, sum_weights)
	sum_roulette = 0
	for i, weight in enumerate(weights):
		if weight < 0:
			return -1
		sum_roulette = sum_roulette + weight
		if (p < sum_roulette):
			return i
	return -1 

def initialize_assn(alpha, n):
	clustering = []
	cluster_assn = []
	for i in range(n):
		#probs = map(lambda x: (len(x) + 0.0) / (i + alpha), clustering)
		#probs.append(alpha / (i+alpha))

		crp_prior = [(len(x) + 0.0) / (i + alpha) for j, x in enumerate(clustering)]
		crp_prior.append(alpha / (i + alpha))
		crp_prior = numpy.array(crp_prior)
		likelihood = likelihood_fn(data, i, clustering, cluster_assn)
		probs = crp_prior * likelihood

		cluster = sample_with_weights(probs)
		if cluster == len(clustering):
			s = set([i])
			clustering.append(s)
		else:
			clustering[cluster].add(i)
		cluster_assn.append(clustering[cluster])
	#print clustering
	return clustering, cluster_assn

def likelihood_fn(data, i, clustering, cluster_assn):
	means = [numpy.mean(data[list(cluster)]) for cluster in clustering]
	means.append(0)	
	stds = [1 for cluster in clustering]
	stds.append(10)
	return stats.norm.pdf(data[i], means, stds)

def gibbs_sampling_crp(alpha, data):
	num_iter = 100
	num_data = len(data)
	clustering, cluster_assn = initialize_assn(alpha, num_data)
	for t in range(num_iter):
		num_new_clusters = 0.0
		for i in range(num_data):
			cluster_assn[i].remove(i)
			if len(cluster_assn[i]) == 0:
				clustering.remove(cluster_assn[i])
			crp_prior = [(len(x) + 0.0) / (num_data - 1 + alpha) for j, x in enumerate(clustering)]
			crp_prior.append(alpha / (num_data - 1 + alpha))
			crp_prior = numpy.array(crp_prior)
			likelihood = likelihood_fn(data, i, clustering, cluster_assn)
			probs = crp_prior * likelihood

			cluster = sample_with_weights(probs)
			if cluster == len(clustering):
				s = set([i])
				clustering.append(s)
				num_new_clusters += 1
			else:
				clustering[cluster].add(i)
			cluster_assn[i] = clustering[cluster]
		if t % 5 == 0 and t > 20:
			#print num_new_clusters
			alpha = num_new_clusters
		#print [numpy.mean(data[list(cluster)]) for cluster in clustering]
	print [len(x) for x in clustering]

import cProfile
if __name__ == '__main__':
	data = numpy.concatenate((stats.norm.rvs(0, 1, size=200), stats.norm.rvs(10,1,size=200), stats.norm.rvs(12, 1, size=200)))
	random.shuffle(data)
	cProfile.run("gibbs_sampling_crp(1.0, data)")
	#print vq.kmeans(data, 3)
