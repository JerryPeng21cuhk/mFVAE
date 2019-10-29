# mixture Factorized variational auto-encoder for Speaker verification
This is a pytorch implementation of mFVAE in the paper: mixture factorization auto-encoder for unsupervised hierarchical deep factorization of speech signal.
Note that we apply reparameterization tricks on both $q(\omega_i|O_i)$ and $q(y_it}|o__{it})$.
mFAE is also implemented in local/mfvae/model.py. You need to revise the corresponding training and extracting files to use mFAE.






