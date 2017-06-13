library(cluster)
require(factoextra)
setwd("~/Downloads/all_mushrooms/Mushrooms/results/species_model")
# One averaged vector per species, with average score per species
species_vectors = read.table("species_vectors_for_clustering.csv", sep="\t", stringsAsFactors=FALSE, header=FALSE)
names = species_vectors['V1']
species_vectors <- species_vectors[,-1]  
dist_matrix = dist(as.matrix(species_vectors))

hier_cluster_model<-hclust(dist_matrix, method="ward.D2")

# SEE MERGES!
hier_cluster_model$merge

plot(hier_cluster_model)

si = silhouette(cutree(hier_cluster_model,250),dist_matrix)
plot(si)

fviz_nbclust(as.matrix(species_vectors), hcut, method = "silhouette",
             hc_method = "complete", k.max=499)

avg_silhouette_widths=numeric()

for (num_clusters in 2:261){
  si = silhouette(cutree(hier_cluster_model, num_clusters), dist_matrix)
  avg_sil = mean(si[,3])
  print(avg_sil)
  avg_silhouette_widths <- c(avg_silhouette_widths, avg_sil)
}
plot(avg_silhouette_widths)