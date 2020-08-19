# KGGen-master
Knowledge Graph Population via a Generative Fashion

"CVAE_GAN_BERT_TransE.py" is the proposed algorithm.

"transformer.py" is a implementation of transformer layer.

"data" is the filefolder which contains experiment data.

The output of our algorithm is put in "computing_results/new". 

"computing_results/ExplainResults_TransX.py" can 
translate the results from id to name. So you can read the results.

"computing_results/NotateGeneratedResults.py" can notate the generated triples as N(new), E(Existed), C(Need to Check), and give statistics

"computing_results/ComputeRankLoss.py" can compute the Label Ranking Average Precision of computing result
