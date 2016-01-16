# lowrank-gru
Gated Recurrent Unit with Low-rank matrix factorization

A reparametrization of the Gated Recurrent Unit (Cho et al. 2014, http://arxiv.org/abs/1406.1078) where the recurrent matrices are constrained to be low-rank. Reduces data complexity and memory usage.

This code is based on the code for the Unitary Evolution Recurrent Neural Networks (Arjovsky et al. 2015 http://arxiv.org/abs/1511.06464) in order to facilitate a direct comparison. Original repository is https://github.com/amarshah/complex_RNN .

Our model performs comparably or better than Unitary Evolution Recurrent Neural Networks on all the tasks we tested (memory, addition and randomly-permuted sequential MNIST) for similar number of parameters. In particular, with an accuracy of 93.0% on permuted MNIST compared to the reported UERNN accuracy of 91.4%, we improve the state of the art for this hard task.
