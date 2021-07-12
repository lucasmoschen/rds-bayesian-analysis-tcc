# Bayesian analysis of respondent-driven sampling with outcome uncertainty

**Completion of course work**

**Course:** Undergraduate in Applied Mathematics 

**College:** [School of Applied Mathematics (FGV)](https://emap.fgv.br/en)

**Advisor:** [Luiz Max Carvalho](https://github.com/maxbiostat)

## Main ideia of the project

Respondent-Driven Sampling (RDS) is a procedure used to sample from
hidden or hard-to-reach populations, such as the populations
of heavy drug users and sex workers. This method works similarly to a
branching process in a network, with two different sources of incentive. 
The first stage is called seed, and after, in each phase, the participants recruit the next ones in their subnet.  

Researchers use this method to estimate the prevalence (proportion of individuals) of some characteristic. In this research, each participant answers a series of questions related to the object of study and other covariates. We consider that the outcome of interest is a binary variable subject to measurement error, that is, it is not possible to be sure about the truth of the answer given. We use the concepts of sensitivity and specificity to deal with this. 

Because of our lack of knowledge about nature itself, it is necessary to
model the uncertainty of these variables and, for that, Bayesian statistics is the indicated study area. The idea, therefore, is to propagate uncertainty about the response of participants through the network of contacts.

Finally, we intend to apply this framework efficiently, in particular,
comparing the Markov chain Monte Carlo algorithms and the nested Laplace Approximation (INLA) and programming them with the help of some programming language such as R, Stan, or Python. 

## Network-based and RDS datasets available

|Data Archive|Author|Description|Located at|
|------------|------|-----------|----------|
|Harvard Dataverse|Khoury, Rana B|Activist refugees from Syria|[Open access](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XKOVUN)|
|Princeton|Centers for Disease Control and Prevention (CDC)|HIV transmission in a community of high-risk heterosexuals|[Login access](https://opr.princeton.edu/archive/P90/)|
|Princeton|Salganik, M.J. et al|Heavy Drug Users in Curitiba, Brazil|[Login access](https://opr.princeton.edu/Archive/NSUM/)|
