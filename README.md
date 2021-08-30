# k-ppd-ert

The k-PPD-ERT framework is based on the Extremely Randomized Trees (ERT) algorithm for learning from distributed structured data. The data is assumed to be horizontally partitioned. To share partial information with the mediator, parties employ a Secure Multiparty Computation (SMC) layer on top of distributed ERT, which is robust to k colluding parties.

For the mediator, the server.py in the Mediator folder should be run. The number of data holer parties, scenario number, and name of data set is asked afterward. For the data holder parties, the client.py in the Data holder party folder should be run. Then, the number of data holer parties, number/ID of the data holder party (starting from 0), scenario number, and name of data set is asked. Finally, after starting parties, the learning process begins.
