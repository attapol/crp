using Distributions

function initialize_assn(alpha, n)
    clustering = {}
    cluster_assn = {}
    for i = 1:n
        crp_prior = zeros(length(clustering) + 1)
        for j = 1:length(clustering)
            crp_prior[j] = length(clustering[j]) / (i + alpha)
        end
        crp_prior[length(clustering) + 1] = alpha / (i + alpha)

        likelihood = likelihood_fn(data, i, clustering, cluster_assn)
		probs = crp_prior .* likelihood
        cluster = rand(Categorical(probs / sum(probs)))
        if cluster == length(clustering) + 1
            s = IntSet(i)
            push!(clustering, s)
        else
            push!(clustering[cluster], i)
        end
        push!(cluster_assn, clustering[cluster])
    end
    return clustering, cluster_assn
end

function likelihood_fn(data, i, clustering, cluster_assn)
    means = zeros(length(clustering) + 1)
    stds = zeros(length(clustering) + 1)
    for j in 1:length(clustering)
        indices = Int64[]
        for x in clustering[j]
            push!(indices, x)
        end
        means[j] = mean(data[indices])
        stds[j] = 1
    end
    means[end] = 0
    stds[end] = 1

    #compute the density
    density = zeros(length(clustering) + 1)
    for j in 1:length(density)
        density[j] = pdf(Normal(means[j], stds[j]), data[i])
        if isnan(density[j])
            density[j] = 0
        end
    end
    return density
end

function print_clustering(clustering)
    for cluster in clustering
        indices = Int64[]
        for x in cluster
            push!(indices, x)
        end
        println("mean ", mean(data[indices]), " num members = " , length(indices))
    end
end

function gibbs_sampling_crp(alpha, data)
    num_iter = 100
    num_data = length(data)
	clustering, cluster_assn = initialize_assn(alpha, num_data)
    for t = 1:num_iter
        num_new_clusters = 0
        for i = 1:num_data
            setdiff!(cluster_assn[i], [i])
            if length(cluster_assn[i]) == 0
                cluster_index = findin(clustering, [cluster_assn[i]])
                splice!(clustering, cluster_index[1]);
            end
            crp_prior = zeros(length(clustering) + 1)
            for j = 1:length(clustering)
                crp_prior[j] = length(clustering[j]) / (num_data - 1 + alpha)
            end
            crp_prior[length(clustering) + 1] = alpha / (num_data - 1 + alpha)
			likelihood = likelihood_fn(data, i, clustering, cluster_assn)
			probs = crp_prior .* likelihood
            cluster = rand(Categorical(probs / sum(probs)))
            if cluster == length(clustering) + 1
                s = IntSet(i)
                push!(clustering, s)
                num_new_clusters += 1
            else
                push!(clustering[cluster], i)
            end
            cluster_assn[i] = clustering[cluster]
        end
        if t % 5 == 0 && t > 20
            alpha = num_new_clusters
        end
    end
    print_clustering(clustering)
end

data = [rand(Normal(0, 1), 200), rand(Normal(10, 1), 200), rand(Normal(12, 1), 200)]
shuffle!(data)
for i in 1:10
    tic()
    gibbs_sampling_crp(1, data)
    println(toc())
end

