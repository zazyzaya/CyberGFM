To reproduce:

First run split_lanl to chop up the dataset and build the maps/edge lists

Then run argus_to_csr to build the tgraphs from the chopped up files

nontemporal_split will break it into the reported tr/va/te partitions randomly, holding out all anomalous edges for the test set

../temporal_split will build the dirty, and ts variants.

scrub_lanl will build the clean variant