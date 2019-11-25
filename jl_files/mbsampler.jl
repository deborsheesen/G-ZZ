using StatsBase, Distributions, ProgressMeter


abstract type sampler_list end

# Define abstract type for mini-batch sampler and some basic methods 
abstract type mbsampler end

struct mbsampler_list <: sampler_list
    d::Int64
    mbs::Array{mbsampler}
end
mbsampler_list(d, gw::mbsampler) = mbsampler_list(d, [gw for i in 1:d])

function gsample(gw::sampler_list, i)
    return gsample(gw.mbs[i])
end

function gsample(gw::sampler_list)
    return [gsample(gw, i) for i in 1:gw.d]
end

# returns the importance weights for the sampler
function get_ubf(gw::sampler_list, mb, i)
    return gw.mbs[i].ubf[mb]
end

function get_weights(gw::sampler_list, mb, i)
    return gw.mbs[i].weights[mb]
end

function get_N(gw::sampler_list, i)
    return gw.mbs[i].N
end

function get_mb_size(gw::sampler_list, i)
    return gw.mbs[i].mb_size
end


# returns the importance weights for the sampler
function get_ubf(gw::mbsampler, mb)
    return gw.ubf[mb]
end

# returns weights for the sampler
function get_weights(gw::mbsampler, mb)
    return gw.weights[mb]
end


function get_N(gw::mbsampler)
    return gw.N
end

function get_mb_size(gw::mbsampler)
    return gw.mb_size
end

# computes a monte carlo estimates of the mean of the function f evaluated on the data points corresponding to the vector x 
function mbs_estimate(gw::mbsampler, f, x)
    mb = gsample(gw)
    return  sum(gw.ubf[mb].*map(f,(x[mb])))
end


#-------------------------------------------------
# Minibatch sampler for uniform subsampling
#-------------------------------------------------
struct umbsampler <: mbsampler
    start::Int64
    N::Int64
    mb_size::Int64
    ubf_scalar::Float64
    weights_scalar::Float64
end

umbsampler(start, N, mb_size) = umbsampler(start, N, mb_size, 1/mb_size, 1/N)

function gsample(gw::umbsampler)
    return gw.start + sample(1:gw.N,gw.mb_size)
end

# returns the importance weights for the sampler
function get_ubf(gw::umbsampler, mb)
    if length(mb) > 1
        return gw.ubf_scalar*ones(length(mb))
    else
        return gw.ubf_scalar
    end
end


# returns weights for the sampler
function get_weights(gw::umbsampler, mb)
    return gw.weights_scalar*ones(length(mb))
end

#-------------------------------------------------
# Minibatch sampler for weighted/importance subsampling 
#-------------------------------------------------
struct wumbsampler <: mbsampler
    N::Int64
    mb_size::Int64
    ubf
    weights
end

wumbsampler(N, mb_size, weights) = wumbsampler(built_wumbsampler(N, mb_size, weights)...)

function built_wumbsampler(N, mb_size, weights)
    weights /= sum(weights)
    ubf =  1.0./(mb_size*N*weights)
    return N, mb_size, ubf, weights
end

function gsample(gw::wumbsampler)
    return sample(1:gw.N, Weights(gw.weights), gw.mb_size)
end

#-------------------------------------------------
# Minibatch sampler for subsampling with sparse weights vectors (in the sense that most weights are the same)
#-------------------------------------------------
struct spwumbsampler <: mbsampler
    N::Int64
    mb_size::Int64
    N_het::Int64
    prob_het::Float64
    weights_het::SparseVector{Float64}
    weight_hom::Float64
    ubf_het::SparseVector{Float64}
    ubf_hom::Float64
    #mb_ubf::Array{Float64}

end

function built_spwumbsampler(N, mb_size, weights_het::SparseVector{Float64}, prob_het)
    nzind = weights_het.nzind
    N_het = length(nzind)
    weights_het_sum = sum(weights_het)
    weights_het /= weights_het_sum #make sure weights_het input is normalized
    weights_het *= prob_het
    weight_hom = (1.0-prob_het)/(N-N_het)
    ubf_het = spzeros(N)
    ubf_het[nzind] =  1.0./(mb_size*N*weights_het[nzind])
    ubf_hom =  1.0./(mb_size*N*weight_hom)

    return N, mb_size, N_het, prob_het, weights_het, weight_hom, ubf_het, ubf_hom 
end
spwumbsampler(N, mb_size, weights_het, prob_het) = spwumbsampler(built_spwumbsampler(N, mb_size, weights_het::SparseVector{Float64}, prob_het)...)

# returns the importance weights for the sampler
function get_ubf(gw::spwumbsampler, mb)
    n = length(mb)
    mb_ubf = Array{Float64}(n)
    for i in 1:n
        if mb[i] in gw.ubf_het.nzind
            mb_ubf[i] =  gw.ubf_het[mb[i]]
        else
            mb_ubf[i] = gw.ubf_hom
        end
    end
    return mb_ubf
end

# returns weights for the sampler
function get_weights(gw::spwumbsampler, mb)
    n = length(mb)
    mb_weights = Array{Float64}(n)
    for i in 1:n
        if mb[i] in gw.weights_het.nzind
            mb_weights[i] =  gw.weights_het[mb[i]]
        else
            mb_weights[i] = gw.weight_hom
        end
    end
    return mb_weights
end

function gsample(gw::spwumbsampler)
    mb = Array{Int64}(gw.mb_size)
    n_het = rand(Binomial(gw.mb_size, gw.prob_het))
    mb[1:n_het] = gw.weights_het.nzind[sample(1:gw.N_het, Weights(gw.weights_het.nzval), n_het)]   
    for i in (n_het+1):gw.mb_size
        i_proposal = sample(1:gw.N,1)[1]
        while (i_proposal in gw.weights_het.nzind)
            i_proposal = sample(1:gw.N,1)[1]
        end
        mb[i] = i_proposal
    end
    return mb
end

"""
function gsample(gw::spwumbsampler)
    mb = zeros(gw.mb_size)
    for i in 1:gw.mb_size
        if rand(Binomial(1,prob_het)) == 1
            mb[i] = gw.weights_het.nzind[sample(1:gw.N_het, Weights(gw.weights_het.nzval), 1)]    
        else
            i_proposal = sample(1:gw.N,1)
            while !(i_proposal in gw.weights_het.nzind)
                i_proposal = sample(1:gw.N,1)
            end
            mb[i] = i_proposal
        end
    end
    return mb
end
"""


#-------------------------------------------------
# Minibatch sampler for grouped weighted/importance subsampling  
#-------------------------------------------------

struct gmbsampler <: mbsampler
    N::Int64
    mb_size::Int64
    ubf
    weights
    labels
    n_labels 
    info_table 
    index_map
    weights_grouped
    weights_grouped_sum
end


function built_gmbsampler(mb_size, weights, grouping) 
    N, = size(weights)
    labels = unique(grouping)
    #print(labels)
    n_labels, = size(labels)
    #print(n_labels)
    info_table = zeros(Int64, n_labels, 4) # group_label | group_size | first index | last index
    separating_index = 1
    index_map = zeros(Int64, size(grouping))
    # build info table
    for (li, l) in enumerate(labels)
        # Set group label
        info_table[li,1] = l
        # Calculate and set group size
        group_size, = size(find(x->x == l, grouping))
        info_table[li,2] = group_size
        # Set start index for group
        info_table[li,3] = separating_index
        # Set end index for group
        info_table[li,4] = separating_index + group_size - 1
        separating_index = separating_index + group_size
    end
    for (li, l) in enumerate(labels)
        index_map[info_table[li,3]:info_table[li,4]] = find(x->x == l, grouping)
    end
    ubf = 1.0./(N * weights*mb_size)
    weights_grouped = zeros(Float64, size(grouping))
    weights_grouped_sum = zeros(Float64, n_labels)
    for (li, l) in enumerate(labels)
        weights_temp = weights[index_map[info_table[li,3]:info_table[li,4]]]
        weights_grouped_sum[li] = sum(weights_temp)
        weights_grouped[info_table[li,3]:info_table[li,4]] = weights_temp/weights_grouped_sum[li]
    end
    return N, mb_size, ubf, weights, labels, n_labels, info_table, index_map, weights_grouped, weights_grouped_sum
end    

gmbsampler(mb_size, weights_a,grouping_a) = gmbsampler(built_gmbsampler(mb_size, weights_a, grouping_a)...)

function gsample(gw::gmbsampler)
    gw.mb_size
    mb = zeros(Int64, gw.mb_size)
    u_groups = sample(1:gw.n_labels, Weights(gw.weights_grouped_sum), gw.mb_size; replace=true)
    #print(u_groups)
    for (li, l) in enumerate(u_groups)
        if gw.info_table[l,1] <= 0
            index = gw.info_table[l,3]:gw.info_table[l,4]
            mb[li] = gw.index_map[sample(index,Weights(gw.weights_grouped[index]),1)...]
        else
            index = gw.info_table[l,3]:gw.info_table[l,4]
            mb[li] = gw.index_map[sample(index,1)...]
        end
    end
    return mb
end

#-------------------------------------------------
# Minibatch sampler list for subsampling using control variates 
#-------------------------------------------------

struct cvmbsampler_list <: sampler_list
    d::Int64
    mbs::Array{mbsampler}
    gradient_log_posterior_root_sum::Array{Float64}
    gradient_log_prior_root::Array{Float64}
    gradient_log_ll_root_vec
    root::Array{Float64}
end
cvmbsampler_list(m, mbs, root) = cvmbsampler_list(build_cvmbsampler_list(m, mbs, root)...)
cvmbsampler_list(m, mbs, root, is_sparse) = cvmbsampler_list(build_cvmbsampler_list(m, mbs, root, is_sparse)...)

function build_cvmbsampler_list(m, mbs, root, is_sparse=false)
    d = length(root)
    Nobs = length(m.ll.y)
#    gradient_log_posterior_root_sum = gradient(m, root) 
    gradient_log_prior_root = gradient(m.pr, root) 
    
    if is_sparse
        gradient_log_ll_root_vec = spzeros(d, Nobs)
        for i in 1:d
            #Compute gradient only for datapoints with non-zeros entry in the i-th dimension (this is fine because partial derivative is zero otherwise for logistic regression) 
            nz_ind = m.ll.X[i,:].nzind 
            gradient_log_ll_root_vec[i,nz_ind] = partial_derivative_vec(m.ll, root, i, nz_ind)
        end
    else
        gradient_log_ll_root_vec = zeros(d, Nobs)
        for i in 1:d
            gradient_log_ll_root_vec[i,:] = partial_derivative_vec(m.ll, root, i, 1:Nobs)
        end
    end
    gradient_log_posterior_root_sum = sum(gradient_log_ll_root_vec,2) + gradient_log_prior_root
    
    return d, mbs, gradient_log_posterior_root_sum, gradient_log_prior_root, gradient_log_ll_root_vec, root
end

#-------------------------------------------------
# Minibatch sampler for stratified subsampling  
#-------------------------------------------------

struct cmbsampler <: mbsampler
    N::Int64
    mb_size::Int64
    ubf
    csamplers::Array{mbsampler}
    clusters 
    n_clusters
    cluster_size 
    mb_table # number of samples per cluster | first index in minibatch| last index in minbatch | 
    weights
end

#function get_weights(gw::cmbsampler, mb)
    #return 1./ (gw.N * gw.ubf[mb])
#    return 1./ (gw.N * gw.ubf[mb])
#end

function built_cmbsampler(csamplers, clusters, weights) 
    n_clusters, = size(csamplers)
    cluster_size = zeros(Int64, n_clusters)
    for ci in 1:n_clusters
        if csamplers[ci].N == size(clusters[ci])[1]
            cluster_size[ci] = csamplers[ci].N
        else
            print("Assigned cluster size and domain size of mb sampler don't match")
        end
    end
    
    N = sum(cluster_size) 
    mb_table = zeros(Int64, n_clusters, 3)
    separating_index = 1
    for (ci, c) in  enumerate(clusters)
        mb_table[ci,1] = csamplers[ci].mb_size
        # Set start index in minibatch for samples from cluster c
        mb_table[ci,2] = separating_index
        # Set end index in minibatch for samples from cluster c
        mb_table[ci,3] = separating_index + csamplers[ci].mb_size - 1
        separating_index = separating_index +  csamplers[ci].mb_size
    end
    
    mb_size = sum(mb_table[:,1])
    ubf = zeros(Float64, N) 
    #weights = zeros(Float64, N) 
    for (ci, c) in  enumerate(clusters)
        #weights_sum = sum(csamplers[ci].weights)
        for (vi, v) in enumerate(c)
            ubf[v] = (cluster_size[ci]/N) * csamplers[ci].ubf[vi] 
            #weights[v] = (csamplers[ci].weights[vi]/weights_sum)*(cluster_size[ci]/N)
        end
    end
    
    
    return N, mb_size, ubf, csamplers, clusters, n_clusters, cluster_size, mb_table, weights
end
cmbsampler(csamplers, clusters, weights) = cmbsampler(built_cmbsampler(csamplers, clusters, weights)...)

function gsample(gw::cmbsampler)
    mb = zeros(Int64, gw.mb_size)
    for (ci, c) in enumerate(gw.clusters)
        index = gw.mb_table[ci,2]:gw.mb_table[ci,3]
        mb_temp = gsample(gw.csamplers[ci])
        mb[index] = c[mb_temp]
    end
    return mb
end

#-------------------------------------------------
# Minibatch sampler for sparse stratified subsampling
#-------------------------------------------------


struct spcmbsampler <: mbsampler
    N::Int64
    mb_size::Int64
    ubf
    csamplers::Array{mbsampler}
    clusters 
    n_clusters
    cluster_size 
    mb_table # number of samples per cluster | first index in minibatch| last index in minbatch | 
    weights
end

#function get_weights(gw::cmbsampler, mb)
    #return 1./ (gw.N * gw.ubf[mb])
#    return 1./ (gw.N * gw.ubf[mb])
#end

function built_spcmbsampler(csamplers, clusters, weights) 
    n_clusters, = size(csamplers)
    cluster_size = zeros(Int64, n_clusters)
    for ci in 1:n_clusters
        if csamplers[ci].N == size(clusters[ci])[1]
            cluster_size[ci] = csamplers[ci].N
        else
            print("Assigned cluster size and domain size of mb sampler don't match")
        end
    end
    
    N = length(weights) # length of weight vector specifies number of observations
    if issparsevec(weights)
        @assert length(weights.nzval) == sum(cluster_size) "Sum of cluster sizes must match number of non-zero elements in weight vector" # Quick sanity check. For a more careful treatment one would need to check that the non-zeros indices for weight vector coincide with the union of the clusters
    end
    
    mb_table = zeros(Int64, n_clusters, 3)
    separating_index = 1
    for (ci, c) in  enumerate(clusters)
        mb_table[ci,1] = csamplers[ci].mb_size
        # Set start index in minibatch for samples from cluster c
        mb_table[ci,2] = separating_index
        # Set end index in minibatch for samples from cluster c
        mb_table[ci,3] = separating_index + csamplers[ci].mb_size - 1
        separating_index = separating_index +  csamplers[ci].mb_size
    end
    
    mb_size = sum(mb_table[:,1])
        
    if issparsevec(weights)
        ubf = spzeros(N) 
    else
        ubf = zeros(Float64, N) 
    end
    #weights = zeros(Float64, N) 
    for (ci, c) in  enumerate(clusters)
        #weights_sum = sum(csamplers[ci].weights)
        for (vi, v) in enumerate(c)
            #ubf[v] = (cluster_size[ci]/N) * csamplers[ci].ubf[vi] 
            ubf[v] = (cluster_size[ci]/N) * get_ubf(csamplers[ci], vi) 
            #weights[v] = (csamplers[ci].weights[vi]/weights_sum)*(cluster_size[ci]/N)
        end
    end    
    
    return N, mb_size, ubf, csamplers, clusters, n_clusters, cluster_size, mb_table, weights
end
spcmbsampler(csamplers, clusters, weights) = spcmbsampler(built_spcmbsampler(csamplers, clusters, weights)...)

function issparsevec(vector)
    return false
end
function issparsevec(vector::AbstractSparseVector)
    return true
end

function gsample(gw::spcmbsampler)
    mb = zeros(Int64, gw.mb_size)
    for (ci, c) in enumerate(gw.clusters)
        index = gw.mb_table[ci,2]:gw.mb_table[ci,3]
        mb_temp = gsample(gw.csamplers[ci])
        mb[index] = c[mb_temp]
    end
    return mb
end


#-------------------------------------------------
# Clustering methods
#-------------------------------------------------

struct cluster_node
    may_have_kids::Bool
    score::Float64
    first_index::Int64
    split_index::Int64

    last_index::Int64
    kid1_score::Float64
    kid2_score::Float64
    delta_score::Float64
end

function greedy_split_cluster(N_cluster, vector)
    #print(length(vector),"\n")
    cluster_node_list = [cluster_node(1,length(vector),vector)]
    for i in 1:(N_cluster-1)
        jmin = -1
        delta_score_min = Inf
        for j in 1:length(cluster_node_list)
            if cluster_node_list[j].may_have_kids && cluster_node_list[j].delta_score <  delta_score_min
                jmin = j
                delta_score_min = cluster_node_list[j].delta_score
            end
        end
        new_cluster_node1, new_cluster_node2 = split(cluster_node_list[jmin], vector)
        deleteat!(cluster_node_list, jmin)
        push!(cluster_node_list, new_cluster_node1, new_cluster_node2)
    end
    return cluster_node_list
end



cluster_node(first_index::Int64,last_index::Int64, vector) =
cluster_node(build_cluster_node(first_index,last_index, vector)...)

cluster_node(first_index::Int64,last_index::Int64, vector, score::Float64) =
cluster_node(build_cluster_node(first_index,last_index, vector, score)...)

function build_cluster_node(first_index::Int64,last_index::Int64, vector, score::Float64)
    may_have_kids = !(first_index == last_index)
    if may_have_kids
        split_index, kid1_score, kid2_score = find_split(first_index,last_index,vector)
        #kid1_score = comp_score(first_index,split_index,vector)
        #kid2_score = comp_score(split_index+1, last_index,vector)
    else
        split_index = -1
        kid1_score = Inf
        kid2_score = Inf
    end
    delta_score = kid1_score + kid2_score - score
    return may_have_kids, score, first_index, split_index, last_index, kid1_score, kid2_score, delta_score
end

function build_cluster_node(first_index::Int64,last_index::Int64, vector)
    #print(length(vector),"\n")
    score = comp_score(first_index, last_index,vector)
    #print("last index: ", last_index, "\n")
    return build_cluster_node(first_index,last_index, vector, score)
end


function comp_score( first_index, last_index,vector)
    #print("last index: ", last_index,"\n")
    Nit = last_index - first_index + 1
    return Nit * (vector[last_index] - vector[first_index])
end
function find_split( first_index, last_index, vector) #comp_split_score_wc_vec #comp_split_score_wc(....,i)
    #print(size(vector))
    Nit = last_index - first_index + 1
    split_score = zeros(Nit-1,2)
    range = first_index:(last_index-1)
    for (i_shift1,i) in enumerate(range)
        #print("size range: ", size(range), "\n")
        #print("size split_score: ", size(split_score), "\n")
        split_score[i_shift1,1] = comp_score(first_index, i,vector) 
        split_score[i_shift1,2] = comp_score(i+1, last_index,vector)
    end
    _, min_i_shift1 = findmin(sum(split_score,2))
    return range[min_i_shift1], split_score[min_i_shift1,1], split_score[min_i_shift1,2]
end

function split(parent_node, vector)
    kid1 = cluster_node(parent_node.first_index, parent_node.split_index, vector, parent_node.kid1_score)
    kid2 = cluster_node(parent_node.split_index+1, parent_node.last_index, vector, parent_node.kid2_score)
    return kid1, kid2
end

function get_clustering(N_cluster, vector_unsorted)
    p = sortperm(vector_unsorted)
    cluster_node_list = greedy_split_cluster(N_cluster,vector_unsorted[p])
    clusters = []
    for (i, node) in enumerate(cluster_node_list)
        #print("c-",i, " : ",node.first_index,"-", node.last_index,"\n")
        clusters =push!(clusters , p[node.first_index:node.last_index])
    end
    return clusters
end