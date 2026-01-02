# NaLBaNA (Natural Language to Bayesian Network Agent)

NaLBaNA is a Python package that allows users to input natural language prompts
the describe causal systems and generate Bayesian networks that represent the 
causal dynamics of system described by those prompts. The generated Bayesian networks represent
"type-level" or "generic" causal relationships, such as "smoking causes lung cancer."

The package assumes that users will have decent familiarity with the theory
of Bayesian networks and the do-calculus. For a concise introduction to Bayesian
networks, see Section 3.1 of Sprites, P. (2010) ["Introduction to Causal Inference"](https://www.jmlr.org/papers/volume11/spirtes10a/spirtes10a.pdf])
JMLR 11: 1643-1662. For a similarly concise introduction to the do-calculus, see Huang, M. and M. Valtorta. (2012)
["Pearl's Calculus of Intervention is Complete"](https://arxiv.org/pdf/1206.6831) Proceedings of the Twenty-Second Conference on Uncertainty in Artificial Intelligence (pp. 217-224).

The core function of the Python package is to allow a user to input a natural-language description of a causal system, and 
receive as output a Python object containing:

1. A set of random variables representing the causal relata of the described system.
2. A set of mutually exclusive and jointly exhausting values for each variable.
3. A directed acyclic graph (or "DAG") representing the causal relationships between 
4. A joint probability distribution over each variable in the DAG.

All four of these steps are accomplished by strategically prompting a large language model 
(or "LLM"), specifically GPT-4.1. As such, _all four of the elements of the Bayeian network 
described above represent an LLM's "best guess" at the underlying causal dynamics of the described
system (i.e., the most likely response given the user input and the LLM's training data), 
rather than any specific "ground truth" or data-based reality._

After generating a Bayesian network from a natural language input, users can obtain joint, conditional, and
marginal probabilities over variables in the network, simulate interventions on combinations of variables in the network,
and change the conditional probability distribution over a variable in the network, given some combination of 
its parents.

NaLBaNA requires the user to input and use [an OpenAI API key](https://openai.com/api/). API costs are only 
incurred when a Bayesian Network is generated from a prompt. Generating a Bayesian network tends to cost between $0.01 and 
$0.05. Ensure appropriate spending warnings and limits are in place before deploying NaLBaNA at scale.
