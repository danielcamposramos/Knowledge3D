Claude, my valued partner, while the foundations are being layd, let's program our next steps.
I need you to dispach several agents for us to prepare what we'll include as knowledge (bi-directionally symlinked to the words and math content), so I also asked to Codex to enhance our ollama_specialists kimi_swarm with internet search for main kimi and also for the agents:
Done.

`kimi_swarm` now supports:
- shared search context: `web_queries`
- Agent A private search context: `sub_task_a_web_queries`
- Agent B private search context: `sub_task_b_web_queries`
- file output: `output_file`

What I changed:
- patched [ollama_specialists.py](/home/daniel/.claude/ollama_specialists.py)
- restarted Docker container `k3d-ollama-mcp`
- verified the running container sees the new parameters in `/app/ollama_specialists.py`

About cleanup:
- there is no persistent browser instance here
- search uses `DDGS()` inside a context manager per call, so each search session is opened and closed immediately
- the fetched search results are just local strings inside the function scope and are discarded when the swarm call returns
- no long-lived page/session state is retained in the container from these searches

So the instance-growth concern is already handled by the implementation model.

New shape:
```python
kimi_swarm(
    task: str,
    sub_task_a: str = "",
    sub_task_b: str = "",
    context: str = "",
    repo_files: str = "",
    web_queries: str = "",
    sub_task_a_web_queries: str = "",
    sub_task_b_web_queries: str = "",
    think: bool = False,
    output_file: str = "",
)
```

Example:
```python
kimi_swarm(
    task="Design a sovereign benchmark ingestion path",
    sub_task_a="Research GPU-side queue patterns",
    sub_task_b="Research canonical multilingual ingestion patterns",
    web_queries="K3D sovereign ingestion architecture",
    sub_task_a_web_queries="CUDA MPSC ring buffer warp aggregated atomics",
    sub_task_b_web_queries="OntoLex canonical lemma graph multilingual symlink design",
    think=True,
    output_file="/K3D/Knowledge3D.local/logs/kimi_swarm_ingestion_research.md",
)
```
----
And these are the topics:
Algebraic modeling languages (AML) are high-level computer programming languages for describing and solving high complexity problems for large scale mathematical computation (i.e. large scale optimization type problems).[1] One particular advantage of some algebraic modeling languages like AIMMS,[1] AMPL,[2] GAMS,[1] Gekko, MathProg, Mosel,[1][3] and OPL is the similarity of their syntax to the mathematical notation of optimization problems. This allows for a very concise and readable definition of problems in the domain of optimization, which is supported by certain language elements like sets, indices, algebraic expressions, powerful sparse index and data handling variables, constraints with arbitrary names. The algebraic formulation of a model does not contain any hints how to process it.

An AML does not solve those problems directly; instead, it calls appropriate external algorithms to obtain a solution. These algorithms are called solvers and can handle certain kind of mathematical problems like:

    linear problems
    integer problems
    (mixed integer) quadratic problems
    mixed complementarity problems
    mathematical programs with equilibrium constraints
    constrained nonlinear systems
    general nonlinear problems
    non-linear programs with discontinuous derivatives
    nonlinear integer problems
    global optimization problems
    stochastic optimization problems

Core elements

The core elements of an AML are:

    a modeling language interpreter (the AML itself)
    solver links
    user interfaces (UI)
    data exchange facilities

Design principles

Most AML follow certain design principles:

    a balanced mix of declarative and procedural elements
    open architecture and interfaces to other systems
    different layers with separation of:
        model and data
        model and solution methods
        model and operating system
        model and interface

Data driven model generation

Most modeling languages exploit the similarities between structured models and relational databases [4] by providing a database access layer, which enables the modelling system to directly access data from external data sources (e.g. these[5] table handlers for AMPL). With the refinement of analytic technologies applied to business processes, optimization models are becoming an integral part of decision support systems; optimization models can be structured and layered to represent and support complex business processes. In such applications, the multi-dimensional data structure typical of OLAP systems can be directly mapped to the optimization models and typical MDDB operations can be translated into aggregation and disaggregation operations on the underlying model [6]
History

Algebraic modelling languages find their roots in matrix-generator and report-writer programs (MGRW), developed in the late seventies. Some of these are MAGEN, MGRW (IBM), GAMMA.3, DATAFORM and MGG/RWG. These systems simplified the communication of problem instances to the solution algorithms and the generation of a readable report of the results.

An early matrix-generator for LP was developed around 1969 at the Mathematisch Centrum (now CWI), Amsterdam.[7] Its syntax was very close to the usual mathematical notation, using subscripts en sigmas. Input for the generator consisted of separate sections for the model and the data. It found users at universities and in industry. The main industrial user was the steel maker Hoogovens (now Tata Steel) where it was used for nearly 25 years.

A big step towards the modern modelling languages is found in UIMP,[8] where the structure of the mathematical programming models taken from real life is analyzed for the first time, to highlight the natural grouping of variables and constraints arising from such models. This led to data-structure features, which supported structured modelling; in this paradigm, all the input and output tables, together with the decision variables, are defined in terms of these structures, in a way comparable to the use of subscripts and sets. This is probably the single most notable feature common to all modern AMLs and enabled, in time, a separation between the model structure and its data, and a correspondence between the entities in an MP model and data in relational databases. So, a model could be finally instantiated and solved over different datasets, just by modifying its datasets.

The correspondence between modelling entities and relational data models,[4] made then possible to seamlessly generate model instances by fetching data from corporate databases. This feature accounts now for a lot of the usability of optimization in real life applications, and is supported by most well-known modelling languages.

While algebraic modelling languages were typically isolated, specialized and commercial languages, more recently algebraic modelling languages started to appear in the form of open-source, specialized libraries within a general-purpose language, like Gekko or Pyomo for Python or JuMP for the Julia language.
Notable AMLs
Specialized AMLs

    AIMMS
    AMPL
    GAMS
    MathProg
    MiniZinc
    OPL

AML Packages in Generic Programming Languages

    FlopC++ for C++
    OptimJ for Java
    JuMP for Julia
    GBOML for Python
    Pyomo for Python
----
In mathematical optimization and computer science, heuristic (from Greek εὑρίσκω eurísko "I find, discover"[1]) is a technique designed for problem solving more quickly when classic methods are too slow for finding an exact or approximate solution, or when classic methods fail to find any exact solution in a search space. This is achieved by trading optimality, completeness, accuracy, or precision for speed. In a way, it can be considered a shortcut.

A heuristic function, also simply called a heuristic, is a function that ranks alternatives in search algorithms at each branching step based on available information to decide which branch to follow. For example, it may approximate the exact solution.[2]
Definition and motivation

The objective of a heuristic is to produce a solution in a reasonable time frame that is good enough for solving the problem at hand. This solution may not be the best of all the solutions to this problem, or it may simply approximate the exact solution. But it is still valuable because finding it does not require a prohibitively long time.

Heuristics may produce results by themselves, or they may be used in conjunction with optimization algorithms to improve their efficiency (e.g., they may be used to generate good seed values).

Results about NP-hardness in theoretical computer science make heuristics the only viable option for a variety of complex optimization problems that need to be routinely solved in real-world applications.

Heuristics underlie the whole field of artificial intelligence and the computer simulation of thinking, as they may be used in situations where there are no known algorithms.[3]
Examples
Simpler problem

One way of achieving the computational performance gain expected of a heuristic consists of solving a simpler problem whose solution is also a solution to the initial problem.
Travelling salesman problem

An example of approximation is described by Jon Bentley for solving the travelling salesman problem (TSP):

    "Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?"

so as to select the order to draw using a pen plotter. TSP is known to be NP-hard so an optimal solution for even a moderate size problem is difficult to solve. Instead, the greedy algorithm can be used to give a good but not optimal solution (it is an approximation to the optimal answer) in a reasonably short amount of time. The greedy algorithm heuristic says to pick whatever is currently the best next step regardless of whether that prevents (or even makes impossible) good steps later. It is a heuristic in the sense that practice indicates it is a good enough solution, while theory indicates that there are better solutions (and even indicates how much better, in some cases).[4]
Search

Another example of heuristic making an algorithm faster occurs in certain search problems. Initially, the heuristic tries every possibility at each step, like the full-space search algorithm. But it can stop the search at any time if the current possibility is already worse than the best solution already found. In such search problems, a heuristic can be used to try good choices first so that bad paths can be eliminated early (see alpha–beta pruning). In the case of best-first search algorithms, such as A* search, the heuristic improves the algorithm's convergence while maintaining its correctness as long as the heuristic is admissible.
Newell and Simon: heuristic search hypothesis

In their Turing Award acceptance speech, Allen Newell and Herbert A. Simon discuss the heuristic search hypothesis: a physical symbol system will repeatedly generate and modify known symbol structures until the created structure matches the solution structure. Each following step depends upon the step before it, thus the heuristic search learns what avenues to pursue and which ones to disregard by measuring how close the current step is to the solution. Therefore, some possibilities will never be generated as they are measured to be less likely to complete the solution.

A heuristic method can accomplish its task by using search trees. However, instead of generating all possible solution branches, a heuristic selects branches more likely to produce outcomes than other branches. It is selective at each decision point, picking branches that are more likely to produce solutions.[5]
Common Heuristic Algorithms in AI

Heuristics are central to many informed search algorithms and optimization techniques for AI:[6]

    A* Search Algorithm The A* search algorithm is one of the most popular heuristic search techniques due to its ability to find optimal solutions efficiently. A* combines both the actual path cost and the heuristic estimate of the remaining cost to reach the goal.

    Greedy Best-First Search: This algorithm expands the node that is closest to the goal based solely on the heuristic function (h(n)), without considering the path cost so far. It is fast but does not guarantee an optimal solution.
    Hill Climbing: A local search algorithm that iteratively moves from the current state to a better neighboring state. It is simple to implement but can get stuck in local optima (suboptimal solutions that are better than their immediate neighbors but not the overall best). An objective function (like a gradient for continuous spaces) to determine direction.
    Simulated Annealing: Simulated Annealing is a heuristic search technique that explores the search space by occasionally accepting worse solutions to avoid getting stuck in local maxima. Inspired by the annealing process in metallurgy, this algorithm gradually reduces the probability of accepting worse solutions as the search progresses. By allowing exploration of suboptimal solutions, simulated annealing can escape local maxima and find a better overall solution. It is commonly used in optimization problems where the search space is large and complex.
    Genetic Algorithms: These are inspired by natural selection, using processes like selection, crossover, and mutation to evolve a population of candidate solutions over generations.
    Ant Colony Optimization: A swarm intelligence method inspired by the way ants find paths to food sources, using artificial "pheromones" to guide the search.

Antivirus software

Antivirus software often uses heuristic rules for detecting viruses and other forms of malware. Heuristic scanning looks for code and/or behavioral patterns common to a class or family of viruses, with different sets of rules for different viruses. If a file or executing process is found to contain matching code patterns and/or to be performing that set of activities, then the scanner infers that the file is infected. The most advanced part of behavior-based heuristic scanning is that it can work against highly randomized self-modifying/mutating (polymorphic) viruses that cannot be easily detected by simpler string scanning methods. Heuristic scanning has the potential to detect future viruses without requiring the virus to be first detected somewhere else, submitted to the virus scanner developer, analyzed, and a detection update for the scanner provided to the scanner's users.
Pitfalls
icon
	
This section does not cite any sources. Please help improve this section by adding citations to reliable sources. Unsourced material may be challenged and removed. (November 2025) (Learn how and when to remove this message)

Some heuristics have a strong underlying theory; they are either derived in a top-down manner from the theory or are arrived at based on either experimental or real world data. Others are just rules of thumb based on real-world observation or experience without even a glimpse of theory. The latter are exposed to a larger number of pitfalls.

When a heuristic is reused in various contexts because it has been seen to "work" in one context, without having been mathematically proven to meet a given set of requirements, it is possible that the current data set does not necessarily represent future data sets (see: overfitting) and that purported "solutions" turn out to be akin to noise.

Statistical analysis can be conducted when employing heuristics to estimate the probability of incorrect outcomes. To use a heuristic for solving a search problem or a knapsack problem, it is necessary to check that the heuristic is admissible. Given a heuristic function h ( v i , v g ) {\displaystyle h(v_{i},v_{g})} meant to approximate the true optimal distance d ⋆ ( v i , v g ) {\displaystyle d^{\star }(v_{i},v_{g})} to the goal node v g {\displaystyle v_{g}} in a directed graph G {\displaystyle G} containing n {\displaystyle n} total nodes or vertices labeled v 0 , v 1 , ⋯ , v n {\displaystyle v_{0},v_{1},\cdots ,v_{n}}, "admissible" means roughly that the heuristic underestimates the cost to the goal or formally that h ( v i , v g ) ≤ d ⋆ ( v i , v g ) {\displaystyle h(v_{i},v_{g})\leq d^{\star }(v_{i},v_{g})} for all ( v i , v g ) {\displaystyle (v_{i},v_{g})} where i , g ∈ [ 0 , 1 , . . . , n ] {\displaystyle {i,g}\in [0,1,...,n]}.

If a heuristic is not admissible, it may never find the goal, either by ending up in a dead end of graph G {\displaystyle G} or by skipping back and forth between two nodes v i {\displaystyle v_{i}} and v j {\displaystyle v_{j}} where i , j ≠ g {\displaystyle {i,j}\neq g}.
Etymology
Wiktionary logo
Look up heuristic in Wiktionary, the free dictionary.

The word "heuristic" came into usage in the early 19th century. It is formed irregularly from the Greek word heuriskein, meaning "to find".[7]
See also

    Constructive heuristic
    Metaheuristic: Methods for controlling and tuning basic heuristic algorithms, usually with usage of memory and learning.
    Matheuristics: Optimization algorithms made by the interoperation of metaheuristics and mathematical programming (MP) techniques.
    Reactive search optimization: Methods using online machine learning principles for self-tuning of heuristics.
----
A constructive heuristic is a type of heuristic method which starts with an empty solution and repeatedly extends the current solution until a complete solution is obtained. It differs from local search heuristics which start with a complete solution and then try to improve the current solution further via local moves. Examples of some famous problems that are solved using constructive heuristics are the flow shop scheduling,[1] the vehicle routing problem[2] and the open shop problem.[3]
----
 In computer science and mathematical optimization, a metaheuristic is a higher-level procedure or heuristic designed to find, generate, tune, or select a heuristic (partial search algorithm) that may provide a sufficiently good solution to an optimization problem or a machine learning problem, especially with incomplete or imperfect information or limited computation capacity.[1][2][3][4] Metaheuristics sample a subset of solutions which is otherwise too large to be completely enumerated or otherwise explored. Metaheuristics may make relatively few assumptions about the optimization problem being solved and so may be usable for a variety of problems.[1][5][6] Their use is always of interest when exact or other (approximate) methods are not available or are not expedient, either because the calculation time is too long or because, for example, the solution provided is too imprecise.

Compared to optimization algorithms and iterative methods, metaheuristics do not guarantee that a globally optimal solution can be found on some class of problems.[4] Many metaheuristics implement some form of stochastic optimization, so that the solution found is dependent on the set of random variables generated.[3] In combinatorial optimization, there are many problems that belong to the class of NP-complete problems and thus can no longer be solved exactly in an acceptable time from a relatively low degree of complexity.[7][8] Metaheuristics then often provide good solutions with less computational effort than approximation methods, iterative methods, or simple heuristics.[4][1] This also applies in the field of continuous or mixed-integer optimization.[1][9][10] As such, metaheuristics are useful approaches for optimization problems.[3] Several books and survey papers have been published on the subject.[3][4][1][11][12] Literature review on metaheuristic optimization,[13] suggested that it was Fred Glover who coined the word metaheuristics.[14]

Most literature on metaheuristics is experimental in nature, describing empirical results based on computer experiments with the algorithms. But some formal theoretical results are also available, often on convergence and the possibility of finding the global optimum.[4][15] Also worth mentioning are the no-free-lunch theorems, which state that there can be no metaheuristic that is better than all others for any given problem.

Especially since the turn of the millennium, many metaheuristic methods have been published with claims of novelty and practical efficacy. While the field also features high-quality research, many of the more recent publications have been of poor quality; flaws include vagueness, lack of conceptual elaboration, poor experiments, and ignorance of previous literature.[16][17]
Properties

These are properties that characterize most metaheuristics:[4]

    Metaheuristics are strategies that guide the search process.
    The goal is to efficiently explore the search space in order to find optimal or near–optimal solutions.
    Techniques which constitute metaheuristic algorithms range from simple local search procedures to complex learning processes.
    Metaheuristic algorithms are approximate and usually non-deterministic.
    Metaheuristics are not problem-specific. However, they were often developed in relation to a problem class such as continuous[18][19] or combinatorial optimization[20] and then generalized later in some cases.[21][22]
    They can draw on domain-specific knowledge in the form of heuristics that are controlled by a higher-level strategy of the metaheuristic.
    They can contain mechanisms that prevent them from getting stuck in certain areas of the search space.
    Modern metaheuristics often use the search history to control the search.

Classification
Euler diagram of the different classifications of metaheuristics[23]

There are a wide variety of metaheuristics[3][1] and a number of properties with respect to which to classify them.[4][24][25][26] The following list is therefore to be understood as an example.
Local search vs. global search

One approach is to characterize the type of search strategy.[4] One type of search strategy is an improvement on simple local search algorithms. A well known local search algorithm is the hill climbing method which is used to find local optimums. However, hill climbing does not guarantee finding global optimum solutions.

Many metaheuristic ideas were proposed to improve local search heuristic in order to find better solutions. Such metaheuristics include simulated annealing, tabu search, iterated local search, variable neighborhood search, and GRASP.[4] These metaheuristics can both be classified as local search-based or global search metaheuristics.

Other global search metaheuristic that are not local search-based are usually population-based metaheuristics. Such metaheuristics include ant colony optimization, evolutionary computation such as genetic algorithm or evolution strategies, particle swarm optimization, rider optimization algorithm[27] and bacterial foraging algorithm.[28]
Single-solution vs. population-based

Another classification dimension is single solution vs population-based searches.[4][12] Single solution approaches focus on modifying and improving a single candidate solution; single solution metaheuristics include simulated annealing, iterated local search, variable neighborhood search, and guided local search.[12] Population-based approaches maintain and improve multiple candidate solutions, often using population characteristics to guide the search; population based metaheuristics include evolutionary computation and particle swarm optimization.[12] Another category of metaheuristics is Swarm intelligence which is a collective behavior of decentralized, self-organized agents in a population or swarm. Ant colony optimization,[29] particle swarm optimization,[12] social cognitive optimization and bacterial foraging algorithm[28] are examples of this category.
Hybridization and memetic algorithms

A hybrid metaheuristic is one that combines a metaheuristic with other optimization approaches, such as algorithms from mathematical programming, constraint programming, and machine learning. Both components of a hybrid metaheuristic may run concurrently and exchange information to guide the search.

On the other hand, Memetic algorithms[30] represent the synergy of evolutionary or any population-based approach with separate individual learning or local improvement procedures for problem search. An example of memetic algorithm is the use of a local search algorithm instead of or in addition to a basic mutation operator in evolutionary algorithms.
Parallel metaheuristics

A parallel metaheuristic is one that uses the techniques of parallel programming to run multiple metaheuristic searches in parallel; these may range from simple distributed schemes to concurrent search runs that interact to improve the overall solution.

With population-based metaheuristics, the population itself can be parallelized by either processing each individual or group with a separate thread or the metaheuristic itself runs on one computer and the offspring are evaluated in a distributed manner per iteration.[31] The latter is particularly useful if the computational effort for the evaluation is considerably greater than that for the generation of descendants. This is the case in many practical applications, especially in simulation-based calculations of solution quality.[32][33]
Nature-inspired and metaphor-based metaheuristics
Main articles: Swarm intelligence and List of metaphor-inspired metaheuristics

A very active area of research is the design of nature-inspired metaheuristics. Many recent metaheuristics, especially evolutionary computation-based algorithms, are inspired by natural systems. Nature acts as a source of concepts, mechanisms and principles for designing of artificial computing systems to deal with complex computational problems. Such metaheuristics include simulated annealing, evolutionary algorithms, ant colony optimization and particle swarm optimization.

A large number of more recent metaphor-inspired metaheuristics have started to attract criticism in the research community for hiding their lack of novelty behind an elaborate metaphor.[16][17][25][34] As a result, a number of renowned scientists of the field have proposed a research agenda for the standardization of metaheuristics in order to make them more comparable, among other things.[35] Another consequence is that the publication guidelines of a number of scientific journals have been adapted accordingly.[36][37][38]
Applications

Most metaheuristics are search methods and when using them, the evaluation function should be subject to greater demands than a mathematical optimization. Not only does the desired target state have to be formulated, but the evaluation should also reward improvements to a solution on the way to the target in order to support and accelerate the search process. The fitness functions of evolutionary or memetic algorithms can serve as an example.

Metaheuristics are used for all types of optimization problems, ranging from continuous through mixed integer problems to combinatorial optimization or combinations thereof.[9][39][40] In combinatorial optimization, an optimal solution is sought over a discrete search-space. An example problem is the travelling salesman problem where the search-space of candidate solutions grows faster than exponentially as the size of the problem increases, which makes an exhaustive search for the optimal solution infeasible.[41][42] Additionally, multidimensional combinatorial problems, including most design problems in engineering[6][43][44][45] such as form-finding and behavior-finding, suffer from the curse of dimensionality, which also makes them infeasible for exhaustive search or analytical methods.

Metaheuristics are also frequently applied to scheduling problems. A typical representative of this combinatorial task class is job shop scheduling, which involves assigning the work steps of jobs to processing stations in such a way that all jobs are completed on time and altogether in the shortest possible time.[5][46] In practice, restrictions often have to be observed, e.g. by limiting the permissible sequence of work steps of a job through predefined workflows[47] and/or with regard to resource utilisation, e.g. in the form of smoothing the energy demand.[48][49] Popular metaheuristics for combinatorial problems include genetic algorithms by Holland et al.,[50] scatter search[51] and tabu search[52] by Glover.

Another large field of application are optimization tasks in continuous or mixed-integer search spaces. This includes, e.g., design optimization[6][53][54] or various engineering tasks.[55][56][57] An example of the mixture of combinatorial and continuous optimization is the planning of favourable motion paths for industrial robots.[58][59]
Metaheuristic Optimization Frameworks

A MOF can be defined as ‘‘a set of software tools that provide a correct and reusable implementation of a set of metaheuristics, and the basic mechanisms to accelerate the implementation of its partner subordinate heuristics (possibly including solution encodings and technique-specific operators), which are necessary to solve a particular problem instance using techniques provided’’.[60]

There are many candidate optimization tools which can be considered as a MOF of varying feature. The following list of 33 MOFs is compared and evaluated in detail in:[60] Comet, EvA2, evolvica, Evolutionary::Algorithm, GAPlayground, jaga, JCLEC, JGAP, jMetal, n-genes, Open Beagle, Opt4j, ParadisEO/EO, Pisa, Watchmaker, FOM, Hypercube, HotFrame, Templar, EasyLocal, iOpt, OptQuest, JDEAL, Optimization Algorithm Toolkit, HeuristicLab, MAFRA, Localizer, GALIB, DREAM, Discropt, MALLBA, MAGMA, and UOF. There have been a number of publications on the support of parallel implementations, which was missing in this comparative study, particularly from the late 10s onwards.[32][33][61][62][63]
Contributions

Many different metaheuristics are in existence and new variants are continually being proposed. Some of the most significant contributions to the field are:

    1952: Robbins and Monro work on stochastic optimization methods.[64]
    1954: Barricelli carries out the first simulations of the evolution process and uses them on general optimization problems.[65]
    1963: Rastrigin proposes random search.[66]
    1965: Matyas proposes random optimization.[67]
    1965: Nelder and Mead propose a simplex heuristic, which was shown by Powell to converge to non-stationary points on some problems.[68]
    1965: Ingo Rechenberg discovers the first Evolution Strategies algorithm.[69]
    1966: Fogel et al. propose evolutionary programming.[70]
    1970: Hastings proposes the Metropolis–Hastings algorithm.[71]
    1970: Cavicchio proposes adaptation of control parameters for an optimizer.[72]
    1970: Kernighan and Lin propose a graph partitioning method, related to variable-depth search and prohibition-based (tabu) search.[73]
    1975: Holland proposes the genetic algorithm.[50]
    1977: Glover proposes scatter search.[51]
    1978: Mercer and Sampson propose a metaplan for tuning an optimizer's parameters by using another optimizer.[74]
    1980: Smith describes genetic programming.[75]
    1983: Kirkpatrick et al. propose simulated annealing.[76]
    1986: Glover proposes tabu search, first mention of the term metaheuristic.[52]
    1989: Moscato proposes memetic algorithms.[30]
    1990: Moscato and Fontanari,[77] and Dueck and Scheuer,[78] independently proposed a deterministic update rule for simulated annealing which accelerated the search. This led to the threshold accepting metaheuristic.
    1992: Dorigo introduces ant colony optimization in his PhD thesis.[29]
    1995: Wolpert and Macready prove the no free lunch theorems.[79][80][81][82]
----
Matheuristics[1][2] are problem agnostic optimization algorithms that make use of mathematical programming (MP) techniques in order to obtain heuristic solutions. Problem-dependent elements are included only within the lower-level mathematic programming, local search or constructive components. An essential feature is the exploitation in some part of the algorithms of features derived from the mathematical model of the problems of interest, thus the definition "model-based heuristics" appearing in the title of some events of the conference series dedicated to matheuristics matheuristics web page.

The topic has attracted the interest of a community of researchers, and this led to the publication of dedicated volumes and journal special issues[3][4][5] besides to dedicated tracks and sessions on wider scope conferences.

A word of caution is needed before delving into the subject, because obviously the use of MP for solving optimization problems, albeit in a heuristic way, is much older and much more widespread than matheuristics. However, this is not the case for metaheuristics. Even the very idea of designing MP methods specifically for heuristic solution has innovative traits, when opposed to exact methods which turn into heuristics when enough computational resources are not available.

Some approaches using MP combined with metaheuristics have begun to appear regularly in the matheuristics literature. This combination can go two-ways, both in MP used to improve or design metaheuristics and in metaheuristics used for improving known MP techniques, even though the first of these two directions is by far more studied.
----
A solver is a piece of mathematical software, possibly in the form of a stand-alone computer program or as a software library, that 'solves' a mathematical problem. A solver takes problem descriptions in some sort of generic form and calculates their solution. In a solver, the emphasis is on creating a program or library that can easily be applied to other problems of similar type.
Solver types

Types of problems with existing dedicated solvers include:

    Linear and non-linear equations. In the case of a single equation, the "solver" is more appropriately called a root-finding algorithm.
    Systems of linear equations.
    Nonlinear systems.
    Systems of polynomial equations, which are a special case of non linear systems, better solved by specific solvers.
    Linear and non-linear optimisation problems
    Systems of ordinary differential equations
    Systems of differential algebraic equations
    Boolean satisfiability problems, including SAT solvers
    Quantified boolean formula solvers[1]
    Constraint satisfaction problems
    Shortest path problems
    Minimum spanning tree problems
    Combinatorial optimization[2]
    Game solvers for problems in game theory[3]
    Three-body problem[4]

The General Problem Solver (GPS) is a particular computer program created in 1957 by Herbert Simon, J. C. Shaw, and Allen Newell intended to work as a universal problem solver, that theoretically can be used to solve every possible problem that can be formalized in a symbolic system, given the right input configuration. It was the first computer program that separated its knowledge of problems (in the form of domain rules) from its strategy of how to solve problems (as a general search engine).

General solvers typically use an architecture similar to the GPS to decouple a problem's definition from the strategy used to solve it. The advantage in this decoupling is that the solver does not depend on the details of any particular problem instance. The strategy utilized by general solvers was based on a general algorithm (generally based on backtracking) with the only goal of completeness. This induces an exponential computational time that dramatically limits their usability. Modern solvers use a more specialized approach that takes advantage of the structure of the problems so that the solver spends as little time as possible backtracking.

For problems of a particular class (e.g., systems of non-linear equations) multiple algorithms are usually available. Some solvers implement multiple algorithms. 
----
In computer science, in particular in knowledge representation and reasoning and metalogic, the area of automated reasoning is dedicated to understanding different aspects of reasoning. The study of automated reasoning helps produce computer programs that allow computers to reason completely, or nearly completely, automatically. Although automated reasoning is considered a sub-field of artificial intelligence, it also has connections with theoretical computer science and philosophy.

The most developed subareas of automated reasoning are automated theorem proving (and the less automated but more pragmatic subfield of interactive theorem proving) and automated proof checking (viewed as guaranteed correct reasoning under fixed assumptions).[citation needed] Extensive work has also been done in reasoning by analogy using induction and abduction.[1]

Other important topics include reasoning under uncertainty and non-monotonic reasoning. An important part of the uncertainty field is that of argumentation, where further constraints of minimality and consistency are applied on top of the more standard automated deduction. John Pollock's OSCAR system is an example of an automated argumentation system that is more specific than being just an automated theorem prover.

Tools and techniques of automated reasoning include the classical logics and calculi, fuzzy logic, Bayesian inference, reasoning with maximal entropy and many less formal ad hoc techniques.

In the 2020s, to enhance the ability of large language models to solve complex problems, AI researchers have designed reasoning language models that can spend additional time on the problem before generating an answer[2] and neuro-symbolic architectures that reason in formal logic in order to prevent hallucinations.[3][4][5]
Early years

The development of formal logic played a big role in the field of automated reasoning, which itself led to the development of artificial intelligence. A formal proof is a proof in which every logical inference has been checked back to the fundamental axioms of mathematics. All the intermediate logical steps are supplied, without exception. No appeal is made to intuition, even if the translation from intuition to logic is routine. Thus, a formal proof is less intuitive and less susceptible to logical errors.[6]

Some consider the Cornell Summer meeting of 1957, which brought together many logicians and computer scientists, as the origin of automated reasoning, or automated deduction.[7] Others say that it began before that with the 1955 Logic Theorist program of Newell, Shaw and Simon, or with Martin Davis’ 1954 implementation of Presburger's decision procedure (which proved that the sum of two even numbers is even).[8]

Automated reasoning, although a significant and popular area of research, went through an "AI winter" in the eighties and early nineties. The field subsequently revived, however. For example, in 2005, Microsoft started using verification technology in many of their internal projects and is planning to include a logical specification and checking language in their 2012 version of Visual C.[7]
Significant contributions

Principia Mathematica was a milestone work in formal logic written by Alfred North Whitehead and Bertrand Russell. Its purpose was to derive all or some of the mathematical expressions, in terms of symbolic logic. Principia Mathematica was initially published in three volumes in 1910, 1912 and 1913.[9] It succeeded The Principles of Mathematics, a 1903 book by Bertrand Russell, in which Russell had presented his famous paradox and argued his thesis that mathematics and logic are identical.

Logic Theorist (LT) was the first ever program developed in 1956 by Allen Newell, Cliff Shaw and Herbert A. Simon to "mimic human reasoning" in proving theorems and was demonstrated on fifty-two theorems from chapter two of Principia Mathematica, proving thirty-eight of them.[10] In addition to proving the theorems, the program found a proof for one of the theorems that was more elegant than the one provided by Whitehead and Russell. After an unsuccessful attempt at publishing their results, Newell, Shaw, and Herbert reported in their publication in 1958, The Next Advance in Operation Research:

        "There are now in the world machines that think, that learn and that create. Moreover, their ability to do these things is going to increase rapidly until (in a visible future) the range of problems they can handle will be co- extensive with the range to which the human mind has been applied."[11] 

Examples of Formal Proofs

    Year 	Theorem 	Proof System 	Formalizer 	Traditional Proof
    1986 	First Incompleteness 	Boyer–Moore 	Shankar[12] 	Gödel
    1990 	Quadratic Reciprocity 	Boyer–Moore 	Russinoff[13] 	Eisenstein
    1996 	Fundamental- of Calculus 	HOL Light 	Harrison 	Henstock
    2000 	Fundamental- of Algebra 	Mizar 	Milewski 	Brynski
    2000 	Fundamental- of Algebra 	Rocq (then: Coq) 	Geuvers et al. 	Kneser
    2004 	Four Color 	Rocq (then: Coq) 	Gonthier 	Robertson et al.
    2004 	Prime Number 	Isabelle 	Avigad et al. 	Selberg-Erdős
    2005 	Jordan Curve 	HOL Light 	Hales 	Thomassen
    2005 	Brouwer Fixed Point 	HOL Light 	Harrison 	Kuhn
    2006 	Flyspeck 1 	Isabelle 	Bauer–Nipkow 	Hales
    2007 	Cauchy Residue 	HOL Light 	Harrison 	Classical
    2008 	Prime Number 	HOL Light 	Harrison 	Analytic proof
    2012 	Feit–Thompson 	Rocq (then: Coq) 	Gonthier et al.[14] 	Bender, Glauberman and Peterfalvi
    2016 	Boolean Pythagorean triples problem 	Formalized as SAT 	Heule et al.[15] 	None 

Proof systems

Boyer-Moore Theorem Prover (NQTHM)
    The design of NQTHM was influenced by John McCarthy and Woody Bledsoe. Started in 1971 at Edinburgh, Scotland, this was a fully automatic theorem prover built using Pure Lisp. The main aspects of NQTHM were:

        the use of Lisp as a working logic.
        the reliance on a principle of definition for total recursive functions.
        the extensive use of rewriting and "symbolic evaluation".
        an induction heuristic based the failure of symbolic evaluation.[16][17]

HOL Light
    Written in OCaml, HOL Light is designed to have a simple and clean logical foundation and an uncluttered implementation. It is essentially another proof assistant for classical higher order logic.[18]

Rocq
    Developed in France, Rocq is another automated proof assistant, which can automatically extract executable programs from specifications, as either Objective CAML or Haskell source code. Properties, programs and proofs are formalized in the same language called the Calculus of Inductive Constructions (CIC).[19]

Applications

Automated reasoning has been most commonly used to build automated theorem provers. Oftentimes, however, theorem provers require some human guidance to be effective and so more generally qualify as proof assistants. In some cases such provers have come up with new approaches to proving a theorem. Logic Theorist is a good example of this. The program came up with a proof for one of the theorems in Principia Mathematica that was more efficient (requiring fewer steps) than the proof provided by Whitehead and Russell. Automated reasoning programs are being applied to solve a growing number of problems in formal logic, mathematics and computer science, logic programming, software and hardware verification, circuit design, and many others. The TPTP (Sutcliffe and Suttner 1998) is a library of such problems that is updated on a regular basis. There is also a competition among automated theorem provers held regularly at the CADE conference (Pelletier, Sutcliffe and Suttner 2002); the problems for the competition are selected from the TPTP library.[20] 
----
In information technology a reasoning system is a software system that generates conclusions from available knowledge using logical techniques such as deduction and induction. Reasoning systems play an important role in the implementation of artificial intelligence and knowledge-based systems.

By the everyday usage definition of the phrase, all computer systems are reasoning systems in that they all automate some type of logic or decision. In typical use in the Information Technology field however, the phrase is usually reserved for systems that perform more complex kinds of reasoning. For example, not for systems that do fairly straightforward types of reasoning such as calculating a sales tax or customer discount but making logical inferences about a medical diagnosis or mathematical theorem. Reasoning systems come in two modes: interactive and batch processing. Interactive systems interface with the user to ask clarifying questions or otherwise allow the user to guide the reasoning process. Batch systems take in all the available information at once and generate the best answer possible without user feedback or guidance.[1]

Reasoning systems have a wide field of application that includes scheduling, business rule processing, problem solving, complex event processing, intrusion detection, predictive analytics, robotics, computer vision, and natural language processing.
History

The first reasoning systems were theorem provers, systems that represent axioms and statements in First Order Logic and then use rules of logic such as modus ponens to infer new statements. Another early type of reasoning system were general problem solvers. These were systems such as the General Problem Solver designed by Newell and Simon. General problem solvers attempted to provide a generic planning engine that could represent and solve structured problems. They worked by decomposing problems into smaller more manageable sub-problems, solving each sub-problem and assembling the partial answers into one final answer. Another example general problem solver was the SOAR family of systems.

In practice these theorem provers and general problem solvers were seldom useful for practical applications and required specialised users with knowledge of logic to utilise. The first practical application of automated reasoning were expert systems. Expert systems focused on much more well defined domains than general problem solving such as medical diagnosis or analyzing faults in an aircraft. Expert systems also focused on more limited implementations of logic. Rather than attempting to implement the full range of logical expressions they typically focused on modus-ponens implemented via IF-THEN rules. Focusing on a specific domain and allowing only a restricted subset of logic improved the performance of such systems so that they were practical for use in the real world and not merely as research demonstrations as most previous automated reasoning systems had been. The engine used for automated reasoning in expert systems were typically called inference engines. Those used for more general logical inferencing are typically called theorem provers.[2]

With the rise in popularity of expert systems many new types of automated reasoning were applied to diverse problems in government and industry. Some such as case-based reasoning were off shoots of expert systems research. Others such as constraint satisfaction algorithms were also influenced by fields such as decision technology and linear programming. Also, a completely different approach, one not based on symbolic reasoning but on a connectionist model has also been extremely productive. This latter type of automated reasoning is especially well suited to pattern matching and signal detection types of problems such as text searching and face matching.
Use of logic

The term reasoning system can be used to apply to just about any kind of sophisticated decision support system as illustrated by the specific areas described below. However, the most common use of the term reasoning system implies the computer representation of logic. Various implementations demonstrate significant variation in terms of systems of logic and formality. Most reasoning systems implement variations of propositional and symbolic (predicate) logic. These variations may be mathematically precise representations of formal logic systems (e.g., FOL), or extended and hybrid versions of those systems (e.g., Courteous logic[3]). Reasoning systems may explicitly implement additional logic types (e.g., modal, deontic, temporal logics). However, many reasoning systems implement imprecise and semi-formal approximations to recognised logic systems. These systems typically support a variety of procedural and semi-declarative techniques in order to model different reasoning strategies. They emphasise pragmatism over formality and may depend on custom extensions and attachments in order to solve real-world problems.

Many reasoning systems employ deductive reasoning to draw inferences from available knowledge. These inference engines support forward reasoning or backward reasoning to infer conclusions via modus ponens. The recursive reasoning methods they employ are termed 'forward chaining' and 'backward chaining', respectively. Although reasoning systems widely support deductive inference, some systems employ abductive, inductive, defeasible and other types of reasoning. Heuristics may also be employed to determine acceptable solutions to intractable problems.

Reasoning systems may employ the closed world assumption (CWA) or open world assumption (OWA). The OWA is often associated with ontological knowledge representation and the Semantic Web. Different systems exhibit a variety of approaches to negation. As well as logical or bitwise complement, systems may support existential forms of strong and weak negation including negation-as-failure and 'inflationary' negation (negation of non-ground atoms). Different reasoning systems may support monotonic or non-monotonic reasoning, stratification and other logical techniques.
Reasoning under uncertainty

Many reasoning systems provide capabilities for reasoning under uncertainty. This is important when building situated reasoning agents which must deal with uncertain representations of the world. There are several common approaches to handling uncertainty. These include the use of certainty factors, probabilistic methods such as Bayesian inference or Dempster–Shafer theory, multi-valued ('fuzzy') logic and various connectionist approaches.[4]
Types of reasoning system

This section provides a non-exhaustive and informal categorisation of common types of reasoning system. These categories are not absolute. They overlap to a significant degree and share a number of techniques, methods and algorithms.
Constraint solvers

Constraint solvers solve constraint satisfaction problems (CSPs). They support constraint programming. A constraint is a which must be met by any valid solution to a problem. Constraints are defined declaratively and applied to variables within given domains. Constraint solvers use search, backtracking and constraint propagation techniques to find solutions and determine optimal solutions. They may employ forms of linear and nonlinear programming. They are often used to perform optimization within highly combinatorial problem spaces. For example, they may be used to calculate optimal scheduling, design efficient integrated circuits or maximise productivity in a manufacturing process.[5]
Theorem provers

Theorem provers use automated reasoning techniques to determine proofs of mathematical theorems. They may also be used to verify existing proofs. In addition to academic use, typical applications of theorem provers include verification of the correctness of integrated circuits, software programs, engineering designs, etc.
Logic programs

Logic programs (LPs) are software programs written using programming languages whose primitives and expressions provide direct representations of constructs drawn from mathematical logic. An example of a general-purpose logic programming language is Prolog. LPs represent the direct application of logic programming to solve problems. Logic programming is characterised by highly declarative approaches based on formal logic, and has wide application across many disciplines.
Rule engines

Rule engines represent conditional logic as discrete rules. Rule sets can be managed and applied separately to other functionality. They have wide applicability across many domains. Many rule engines implement reasoning capabilities. A common approach is to implement production systems to support forward or backward chaining. Each rule ('production') binds a conjunction of predicate clauses to a list of executable actions.

At run-time, the rule engine matches productions against facts and executes ('fires') the associated action list for each match. If those actions remove or modify any facts, or assert new facts, the engine immediately re-computes the set of matches. Rule engines are widely used to model and apply business rules, to control decision-making in automated processes and to enforce business and technical policies.
Deductive classifier

Deductive classifiers arose slightly later than rule-based systems and were a component of a new type of artificial intelligence knowledge representation tool known as frame languages. A frame language describes the problem domain as a set of classes, subclasses, and relations among the classes. It is similar to the object-oriented model. Unlike object-oriented models however, frame languages have a formal semantics based on first order logic.

They utilise this semantics to provide input to the deductive classifier. The classifier in turn can analyze a given model (known as an ontology) and determine if the various relations described in the model are consistent. If the ontology is not consistent the classifier will highlight the declarations that are inconsistent. If the ontology is consistent the classifier can then do further reasoning and draw additional conclusions about the relations of the objects in the ontology.

For example, it may determine that an object is actually a subclass or instance of additional classes as those described by the user. Classifiers are an important technology in analyzing the ontologies used to describe models in the Semantic web.[6][7]
Machine learning systems

Machine learning systems evolve their behavior over time based on experience. This may involve reasoning over observed events or example data provided for training purposes. For example, machine learning systems may use inductive reasoning to generate hypotheses for observed facts. Learning systems search for generalised rules or functions that yield results in line with observations and then use these generalisations to control future behavior.
Case-based reasoning systems

Case-based reasoning (CBR) systems provide solutions to problems by analysing similarities to other problems for which known solutions already exist. Case-based reasoning uses the top (superficial) levels of similarity; namely, the object, feature, and value criteria. This differs case-based reasoning from analogical reasoning in that analogical reasoning uses only the "deep" similarity criterion i.e. relationship or even relationships of relationships, and need not find similarity on the shallower levels. This difference makes case-based reasoning applicable only among cases of the same domain because similar objects, features, and/or values must be in the same domain, while the "deep" similarity criterion of "relationships" makes analogical reasoning applicable cross-domains where only the relationships ae similar between the cases. CBR systems are commonly used in customer/technical support and call centre scenarios and have applications in industrial manufacture, agriculture, medicine, law and many other areas.
Procedural reasoning systems
A procedural reasoning system (PRS) uses reasoning techniques to select plans from a procedural knowledge base. Each plan represents a course of action for achievement of a given goal. The PRS implements a belief–desire–intention model by reasoning over facts ('beliefs') to select appropriate plans ('intentions') for given goals ('desires'). Typical applications of PRS include management, monitoring and fault detection systems. 
----
A semantic reasoner, reasoning engine, rules engine, or simply a reasoner, is a piece of software able to infer logical consequences from a set of asserted facts or axioms. The notion of a semantic reasoner generalizes that of an inference engine, by providing a richer set of mechanisms to work with. The inference rules are commonly specified by means of an ontology language, and often a description logic language. Many reasoners use first-order predicate logic to perform reasoning; inference commonly proceeds by forward chaining and backward chaining. There are also examples of probabilistic reasoners, including non-axiomatic reasoning systems,[1] and probabilistic logic networks.[2]
Applications

Notable semantic reasoners and related software:
Free to use (closed source)

    Cyc inference engine, a forward and backward chaining inference engine with numerous specialized modules for high-order logic.
    KAON2 is an infrastructure for managing OWL-DL, SWRL, and F-Logic ontologies.

Free software (open source)

    Cwm, a forward-chaining reasoner used for querying, checking, transforming and filtering information. Its core language is RDF, extended to include rules, and it uses RDF/XML or N3 serializations as required.
    Drools, a forward-chaining inference-based rules engine which uses an enhanced implementation of the Rete algorithm.
    Evrete, a forward-chaining Java rule engine that uses the Rete algorithm and is compliant with the Java Rule Engine API (JSR 94).
    EYE, a reasoning engine performing forward- and backward-chaining along Euler paths, supporting the Semantic Web Stack and implementing Notation3.
    D3web, a platform for knowledge-based systems (expert systems).
    Flora-2, an object-oriented, rule-based knowledge-representation and reasoning system.
    Jena, an open-source semantic-web framework for Java which includes a number of different semantic-reasoning modules.
    OWLSharp, a lightweight and friendly .NET library for realizing intelligent Semantic Web applications.
    NRules a forward-chaining inference-based rules engine implemented in C# which uses an enhanced implementation of the Rete algorithm
    Prova, a semantic-web rule engine which supports data integration via SPARQL queries and type systems (RDFS, OWL ontologies as type system).
    DIP, Defeasible-Inference Platform (DIP) is a Web Ontology Language reasoner and Protégé desktop plugin for representing and reasoning with defeasible subsumption.[3] It implements a Preferential entailment style of reasoning that reduces to "classical entailment" i.e., without the need to modify the underlying decision procedure.

Semantic Reasoner for Internet of Things (open-source)

S-LOR (Sensor-based Linked Open Rules) semantic reasoner S-LOR is under GNU GPLv3 license.

S-LOR (Sensor-based Linked Open Rules) is a rule-based reasoning engine and an approach for sharing and reusing interoperable rules to deduce meaningful knowledge from sensor measurements. 
----
Case-based reasoning (CBR), broadly construed, is the process of solving new problems based on the solutions of similar past problems.[1][2]

In everyday life, an auto mechanic who fixes an engine by recalling another car that exhibited similar symptoms is using case-based reasoning. A lawyer who advocates a particular outcome in a trial based on legal precedents or a judge who creates case law is using case-based reasoning. So, too, an engineer copying working elements of nature (practicing biomimicry) is treating nature as a database of solutions to problems. Case-based reasoning is a prominent type of analogy solution making.

It has been argued[by whom?] that case-based reasoning is not only a powerful method for computer reasoning, but also a pervasive behavior in everyday human problem solving; or, more radically, that all reasoning is based on past cases personally experienced. This view is related to prototype theory, which is most deeply explored in cognitive science.
Process
A diagram of case-based reasoning in French.
A diagram of case-based reasoning in French

Case-based reasoning has been formalized[clarification needed] for purposes of computer reasoning as a four-step process:[3]

    Retrieve: Given a target problem, retrieve cases relevant to solving it from memory. A case consists of a problem, its solution, and, typically, annotations about how the solution was derived. For example, suppose Fred wants to prepare blueberry pancakes. Being a novice cook, the most relevant experience he can recall is one in which he successfully made plain pancakes. The procedure he followed for making the plain pancakes, together with justifications for decisions made along the way, constitutes Fred's retrieved case.
    Reuse: Map the solution from the previous case to the target problem. This may involve adapting the solution as needed to fit the new situation. In the pancake example, Fred must adapt his retrieved solution to include the addition of blueberries.
    Revise: Having mapped the previous solution to the target situation, test the new solution in the real world (or a simulation) and, if necessary, revise. Suppose Fred adapted his pancake solution by adding blueberries to the batter. After mixing, he discovers that the batter has turned blue – an undesired effect. This suggests the following revision: delay the addition of blueberries until after the batter has been ladled into the pan.
    Retain: After the solution has been successfully adapted to the target problem, store the resulting experience as a new case in memory. Fred, accordingly, records his new-found procedure for making blueberry pancakes, thereby enriching his set of stored experiences, and better preparing him for future pancake-making demands.

Comparison to other methods
icon
	
This section needs additional citations for verification. Please help improve this article by adding citations to reliable sources in this section. Unsourced material may be challenged and removed. (March 2016) (Learn how and when to remove this message)

At first glance, CBR may seem similar to the rule induction algorithms[note 1] of machine learning. Like a rule-induction algorithm, CBR starts with a set of cases or training examples; it forms generalizations of these examples, albeit implicit ones, by identifying commonalities between a retrieved case and the target problem.[4]

If for instance a procedure for plain pancakes is mapped to blueberry pancakes, a decision is made to use the same basic batter and frying method, thus implicitly generalizing the set of situations under which the batter and frying method can be used. The key difference, however, between the implicit generalization in CBR and the generalization in rule induction lies in when the generalization is made. A rule-induction algorithm draws its generalizations from a set of training examples before the target problem is even known; that is, it performs eager generalization.

For instance, if a rule-induction algorithm were given recipes for plain pancakes, Dutch apple pancakes, and banana pancakes as its training examples, it would have to derive, at training time, a set of general rules for making all types of pancakes. It would not be until testing time that it would be given, say, the task of cooking blueberry pancakes. The difficulty for the rule-induction algorithm is in anticipating the different directions in which it should attempt to generalize its training examples. This is in contrast to CBR, which delays (implicit) generalization of its cases until testing time – a strategy of lazy generalization. In the pancake example, CBR has already been given the target problem of cooking blueberry pancakes; thus it can generalize its cases exactly as needed to cover this situation. CBR therefore tends to be a good approach for rich, complex domains in which there are myriad ways to generalize a case.

In law, there is often explicit delegation of CBR to courts, recognizing the limits of rule based reasons: limiting delay, limited knowledge of future context, limit of negotiated agreement, etc. While CBR in law and cognitively inspired CBR have long been associated, the former is more clearly an interpolation of rule based reasoning, and judgment, while the latter is more closely tied to recall and process adaptation. The difference is clear in their attitude toward error and appellate review.

Another name for case-based reasoning in problem solving is symptomatic strategies. It does require à priori domain knowledge that is gleaned from past experience which established connections between symptoms and causes. This knowledge is referred to as shallow, compiled, evidential, history-based as well as case-based knowledge. This is the strategy most associated with diagnosis by experts. Diagnosis of a problem transpires as a rapid recognition process in which symptoms evoke appropriate situation categories.[5] An expert knows the cause by virtue of having previously encountered similar cases. Case-based reasoning is the most powerful strategy, and that used most commonly. However, the strategy won't work independently with truly novel problems, or where deeper understanding of whatever is taking place is sought.

An alternative approach to problem solving is the topographic strategy which falls into the category of deep reasoning. With deep reasoning, in-depth knowledge of a system is used. Topography in this context means a description or an analysis of a structured entity, showing the relations among its elements.[6]

Also known as reasoning from first principles,[7] deep reasoning is applied to novel faults when experience-based approaches aren't viable. The topographic strategy is therefore linked to à priori domain knowledge that is developed from a more a fundamental understanding of a system, possibly using first-principles knowledge. Such knowledge is referred to as deep, causal or model-based knowledge.[8] Hoc and Carlier[9] noted that symptomatic approaches may need to be supported by topographic approaches because symptoms can be defined in diverse terms. The converse is also true – shallow reasoning can be used abductively to generate causal hypotheses, and deductively to evaluate those hypotheses, in a topographical search.
Criticism

Critics of CBR[who?] argue that it is an approach that accepts anecdotal evidence as its main operating principle. Without statistically relevant data for backing and implicit generalization, there is no guarantee that the generalization is correct. However, all inductive reasoning where data is too scarce for statistical relevance is inherently based on anecdotal evidence.
History

CBR traces its roots to the work of Roger Schank and his students at Yale University in the early 1980s. Schank's model of dynamic memory[10] was the basis for the earliest CBR systems: Janet Kolodner's CYRUS[11] and Michael Lebowitz's IPP.[12]

Other schools of CBR and closely allied fields emerged in the 1980s, which directed at topics such as legal reasoning, memory-based reasoning (a way of reasoning from examples on massively parallel machines), and combinations of CBR with other reasoning methods. In the 1990s, interest in CBR grew internationally, as evidenced by the establishment of an International Conference on Case-Based Reasoning in 1995, as well as European, German, British, Italian, and other CBR workshops[which?].

CBR technology has resulted in the deployment of a number of successful systems, the earliest being Lockheed's CLAVIER,[13] a system for laying out composite parts to be baked in an industrial convection oven. CBR has been used extensively in applications such as the Compaq SMART system[14] and has found a major application area in the health sciences,[15] as well as in structural safety management.

There is recent work[which?][when?] that develops CBR within a statistical framework and formalizes case-based inference as a specific type of probabilistic inference. Thus, it becomes possible to produce case-based predictions equipped with a certain level of confidence.[16] One description of the difference between CBR and induction from instances is that statistical inference aims to find what tends to make cases similar while CBR aims to encode what suffices to claim similarly.[17][full citation needed] 
----
Casuistry (/ˈkæzjuɪstri/ KAZ-ew-iss-tree) is a process of reasoning for resolving an ethical dilemma (moral problem) either by extracting or by extending abstract rules from a particular case of conscience, and reapplying those abstract rules to other, different ethical dilemmas.[1] Casuistry is a method of reasoning common to applied ethics and jurisprudence. Moreover, in philosophy, the term casuistry is a pejorative criticism of the use of clever but unsound reasoning, especially in ethical questions, as in the case of sophistry.[2] As a method of reasoning, casuistry is both the:

    Study of cases of conscience and a method of solving conflicts of obligations by applying general principles of ethics, religion, and moral theology to particular and concrete cases of human conduct. This frequently demands an extensive knowledge of natural law and equity, civil law, ecclesiastical precepts, and an exceptional skill in interpreting these various norms of conduct ...[3] 

Etymology

The term casuistry and the noun "casuist" date from 1600 and derive from the Latin noun casus, case, as used in the phrase a "case of conscience", and the usual sense of the usage was pejorative.[4]
History

Casuistry dates from Aristotle (384–322 BC), and the peak of casuistry was from 1550 to 1650, when the Society of Jesus (the Jesuits) used casuistic reasoning, particularly in administering the Sacrament of Penance (or "confession").[5] The term became pejorative following Blaise Pascal's attack on the misuse of the method in his Provincial Letters (1656–57).[6] The French mathematician, religious philosopher and Jansenist sympathiser attacked priests who used casuistic reasoning in confession to pacify wealthy church donors. Pascal charged that "remorseful" aristocrats could confess a sin one day, re-commit it the next, then generously donate to the church and return to re-confess their sin, confident that they were being assigned a penance in name only. These criticisms darkened casuistry's reputation in the following centuries. For example, the Oxford English Dictionary quotes a 1738 essay[7] by Henry St. John, 1st Viscount Bolingbroke, to the effect that casuistry "destroys, by distinctions and exceptions, all morality, and effaces the essential difference between right and wrong, good and evil".[8]

The 20th century saw a revival of interest in casuistry. In their book The Abuse of Casuistry: A History of Moral Reasoning (1988), Albert Jonsen and Stephen Toulmin[9] argue that it is not casuistry but its abuse that has been a problem; that, properly used, casuistry is powerful reasoning. Jonsen and Toulmin offer casuistry as a method for compromising the contradictory principles of moral absolutism and moral relativism. In addition, the ethical philosophies of utilitarianism (especially preference utilitarianism) and pragmatism have been identified as employing casuistic reasoning.[by whom?]
Early modernity

The casuistic method was popular among Catholic thinkers in the early modern period. Casuistic authors include Antonio Escobar y Mendoza, whose Summula casuum conscientiae (1627) enjoyed great success, Thomas Sanchez, Vincenzo Filliucci (Jesuit and penitentiary at St Peter's), Antonino Diana, Paul Laymann (Theologia Moralis, 1625), John Azor (Institutiones Morales, 1600), Etienne Bauny, Louis Cellot, Valerius Reginaldus, and Hermann Busembaum (d. 1668).[10]

The progress of casuistry was interrupted toward the middle of the 17th century by the controversy which arose concerning the doctrine of probabilism, which effectively stated that one could choose to follow a "probable opinion"—that is, an opinion supported by a theologian or another—even if it contradicted a more probable opinion or a quotation from one of the Fathers of the Church.[11]

Certain kinds of casuistry were criticised by early Protestant theologians, because it was used to justify many of the abuses that they sought to reform. It was famously attacked by the Catholic and Jansenist philosopher Blaise Pascal during the formulary controversy against the Jesuits, in his Provincial Letters, as the use of rhetorics to justify moral laxity, which became identified by the public with Jesuitism; hence the everyday use of the term to mean complex and sophistic reasoning to justify moral laxity.[12] By the mid-18th century, "casuistry" had become a synonym for attractive-sounding, but ultimately false, moral reasoning.[13]

In 1679 Pope Innocent XI publicly condemned sixty-five of the more radical propositions (stricti mentalis), taken chiefly from the writings of Escobar, Suarez and other casuists as propositiones laxorum moralistarum and forbade anyone to teach them under penalty of excommunication.[14] Despite this condemnation by a pope, both Catholicism and Protestantism permit the use of ambiguous statements in specific circumstances.[15]
Later modernity

G. E. Moore dealt with casuistry in chapter 1.4 of his Principia Ethica, in which he claimed that "the defects of casuistry are not defects of principle; no objection can be taken to its aim and object. It has failed only because it is far too difficult a subject to be treated adequately in our present state of knowledge". Furthermore, he asserted that "casuistry is the goal of ethical investigation. It cannot be safely attempted at the beginning of our studies, but only at the end".[16]

Since the 1960s, applied ethics has revived the ideas of casuistry in applying moral reasoning to particular cases in law, bioethics, and business ethics. Its facility for dealing with situations where rules or values conflict with each other has made it a useful approach in professional ethics, and casuistry's reputation has improved somewhat as a result.[17]

Pope Francis, a Jesuit, criticized casuistry as "the practice of setting general laws on the basis of exceptional cases" in instances where a more holistic approach would be preferred.[18] 
----
Abductive reasoning (also called abduction,[1] abductive inference,[1] or retroduction[2]) is a form of logical inference that seeks the simplest and most likely conclusion from a set of observations. It was formulated and advanced by the American philosopher and logician Charles Sanders Peirce beginning in the latter half of the 19th century.

Abductive reasoning, unlike deductive reasoning, yields a plausible conclusion but does not definitively verify it. Abductive conclusions do not eliminate uncertainty or doubt, which is expressed in terms such as "best available" or "most likely". While inductive reasoning draws general conclusions that apply to many situations, abductive conclusions are confined to the particular observations in question.

In the 1990s, as computing power grew, the fields of law,[3] computer science, and artificial intelligence research[4] spurred renewed interest in the subject of abduction.[5] Diagnostic expert systems frequently employ abduction.[6]
Deduction, induction, and abduction
Main article: Logical reasoning
Deduction
Main article: Deductive reasoning

Deductive reasoning allows deriving b {\displaystyle b} from a {\displaystyle a} only where b {\displaystyle b} is a formal logical consequence of a {\displaystyle a}. In other words, deduction derives the consequences of the assumed. Given the truth of the assumptions, a valid deduction guarantees the truth of the conclusion. For example, given that "Socrates is a man" ( a 1 {\displaystyle a_{1}}) and "All men are mortal" ( a 2 {\displaystyle a_{2}}), it follows that "Socrates is mortal" ( b {\displaystyle b}).
Induction
Main article: Inductive reasoning

Inductive reasoning is the process of inferring some general principle b {\displaystyle b} from a body of knowledge a {\displaystyle a}, where b {\displaystyle b} does not necessarily follow from a {\displaystyle a}. a {\displaystyle a} might give us very good reason to accept b {\displaystyle b} but does not ensure b {\displaystyle b}.

For example, if all swans that a person has observed so far are white, they may infer a universal categorical proposition of the form All swans are white. They have good reason to believe the conclusion from the premise because it is the best explanation for their observations, but the truth of the conclusion is not guaranteed. Indeed, it turns out that some swans are black.[7]
Abduction

Abductive reasoning allows inferring a {\displaystyle a} as an explanation of b {\displaystyle b}. As a result of this inference, abduction allows the precondition a {\displaystyle a} to be abducted from the consequence b {\displaystyle b}. Deductive reasoning and abductive reasoning differ in which end, left or right, of the proposition " a {\displaystyle a} entails b {\displaystyle b}" serves as the conclusion. For example, with deductive reasoning, knowing that it rained heavily during the night you could deduce that the lawn will be wet in the morning, without looking outside. With abductive reasoning, a couple leaving their house in the morning and seeing that their lawn is wet might abduce that it rained while they were asleep. This serves as a hypothesis that "best explains" their observation. Given the many possible explanations for the lawn getting wet, their abduction does not establish certainty that it rained overnight, but it is still useful and can serve to orient them in their surroundings. Despite many possible explanations for any physical process we observe, we tend to abduce a single explanation (or a few) for this process, in the expectation that we can better orient ourselves in our surroundings and disregard some possibilities. Properly used, abductive reasoning can be a useful source of priors in Bayesian statistics.

One can understand abductive reasoning as inference to the best explanation,[8] although the terms abduction and inference to the best explanation are not always used equivalently.[9][10]
Formalizations of abduction
Logic-based abduction

In logic, explanation is accomplished through the use of a logical theory T {\displaystyle T} representing a domain and a set of observations O {\displaystyle O}. Abduction is the process of deriving a set of explanations of O {\displaystyle O} according to T {\displaystyle T} and picking out one of those explanations. For E {\displaystyle E} to be an explanation of O {\displaystyle O} according to T {\displaystyle T}, it should satisfy two conditions:

    O {\displaystyle O} follows from E {\displaystyle E} and T {\displaystyle T};
    E {\displaystyle E} is consistent with T {\displaystyle T}.

In formal logic, O {\displaystyle O} and E {\displaystyle E} are assumed to be sets of literals. The two conditions for E {\displaystyle E} being an explanation of O {\displaystyle O} according to theory T {\displaystyle T} are formalized as:

    T ∪ E ⊨ O ; {\displaystyle T\cup E\models O;}
    T ∪ E {\displaystyle T\cup E} is consistent.

Among the possible explanations E {\displaystyle E} satisfying these two conditions, some other condition of minimality is usually imposed to avoid irrelevant facts (not contributing to the entailment of O {\displaystyle O}) being included in the explanations. Abduction is then the process that picks out some member of E {\displaystyle E}. Criteria for picking out a member representing "the best" explanation include the simplicity, the prior probability, or the explanatory power of the explanation.

A proof-theoretical abduction method for first-order classical logic based on the sequent calculus and a dual one, based on semantic tableaux (analytic tableaux) have been proposed.[11] The methods are sound and complete and work for full first-order logic, without requiring any preliminary reduction of formulae into normal forms. These methods have also been extended to modal logic.[12]

Abductive logic programming is a computational framework that extends normal logic programming with abduction. It separates the theory T {\displaystyle T} into two components, one of which is a normal logic program, used to generate E {\displaystyle E} by means of backward reasoning, the other of which is a set of integrity constraints, used to filter the set of candidate explanations.
Set-cover abduction

A different formalization of abduction is based on inverting the function that calculates the visible effects of the hypotheses. Formally, we are given a set of hypotheses H {\displaystyle H} and a set of manifestations M {\displaystyle M}; they are related by the domain knowledge, represented by a function e {\displaystyle e} that takes as an argument a set of hypotheses and gives as a result the corresponding set of manifestations. In other words, for every subset of the hypotheses H ′ ⊆ H {\displaystyle H'\subseteq H}, their effects are known to be e ( H ′ ) {\displaystyle e(H')}.

Abduction is performed by finding a set H ′ ⊆ H {\displaystyle H'\subseteq H} such that M ⊆ e ( H ′ ) {\displaystyle M\subseteq e(H')}. In other words, abduction is performed by finding a set of hypotheses H ′ {\displaystyle H'} such that their effects e ( H ′ ) {\displaystyle e(H')} include all observations M {\displaystyle M}.

A common assumption is that the effects of the hypotheses are independent, that is, for every H ′ ⊆ H {\displaystyle H'\subseteq H}, it holds that e ( H ′ ) = ⋃ h ∈ H ′ e ( { h } ) {\displaystyle e(H')=\bigcup _{h\in H'}e(\{h\})}. If this condition is met, abduction can be seen as a form of set covering.
Abductive validation

Abductive validation is the process of validating a given hypothesis through abductive reasoning. This can also be called reasoning through successive approximation.[citation needed] Under this principle, an explanation is valid if it is the best possible explanation of a set of known data. The best possible explanation is often defined in terms of simplicity and elegance (see Occam's razor). Abductive validation is common practice in hypothesis formation in science; moreover, Peirce claims that it is a ubiquitous aspect of thought:

    Looking out my window this lovely spring morning, I see an azalea in full bloom. No, no! I don't see that; though that is the only way I can describe what I see. That is a proposition, a sentence, a fact; but what I perceive is not proposition, sentence, fact, but only an image, which I make intelligible in part by means of a statement of fact. This statement is abstract; but what I see is concrete. I perform an abduction when I so much as express in a sentence anything I see. The truth is that the whole fabric of our knowledge is one matted felt of pure hypothesis confirmed and refined by induction. Not the smallest advance can be made in knowledge beyond the stage of vacant staring, without making an abduction at every step.[13]

It was Peirce's own maxim that "Facts cannot be explained by a hypothesis more extraordinary than these facts themselves; and of various hypotheses the least extraordinary must be adopted."[14] After obtaining possible hypotheses that may explain the facts, abductive validation is a method for identifying the most likely hypothesis that should be adopted.
Subjective logic abduction

Subjective logic generalises probabilistic logic by including degrees of epistemic uncertainty in the input arguments, i.e. instead of probabilities, the analyst can express arguments as subjective opinions. Abduction in subjective logic is thus a generalization of probabilistic abduction described above.[15] The input arguments in subjective logic are subjective opinions which can be binomial when the opinion applies to a binary variable or multinomial when it applies to an n-ary variable. A subjective opinion thus applies to a state variable X {\displaystyle X} which takes its values from a domain X {\displaystyle \mathbf {X} } (i.e. a state space of exhaustive and mutually disjoint state values x {\displaystyle x}), and is denoted by the tuple ω X = ( b X , u X , a X ) {\displaystyle \omega _{X}=(b_{X},u_{X},a_{X})\,\!}, where b X {\displaystyle b_{X}\,\!} is the belief mass distribution over X {\displaystyle \mathbf {X} }, u X {\displaystyle u_{X}\,\!} is the epistemic uncertainty mass, and a X {\displaystyle a_{X}\,\!} is the base rate distribution over X {\displaystyle \mathbf {X} }. These parameters satisfy u X + ∑ b X ( x ) = 1 {\displaystyle u_{X}+\sum b_{X}(x)=1\,\!} and ∑ a X ( x ) = 1 {\displaystyle \sum a_{X}(x)=1\,\!} as well as b X ( x ) , u X , a X ( x ) ∈ [ 0 , 1 ] {\displaystyle b_{X}(x),u_{X},a_{X}(x)\in [0,1]\,\!}.

Assume the domains X {\displaystyle \mathbf {X} } and Y {\displaystyle \mathbf {Y} } with respective variables X {\displaystyle X} and Y {\displaystyle Y}, the set of conditional opinions ω X ∣ Y {\displaystyle \omega _{X\mid Y}} (i.e. one conditional opinion for each value y {\displaystyle y}), and the base rate distribution a Y {\displaystyle a_{Y}}. Based on these parameters, the subjective Bayes' theorem denoted with the operator ϕ ~ {\displaystyle \;{\widetilde {\phi }}} produces the set of inverted conditionals ω Y ∣ ~ X {\displaystyle \omega _{Y{\tilde {\mid }}X}} (i.e. one inverted conditional for each value x {\displaystyle x}) expressed by:

ω Y | ~ X = ω X | Y ϕ ~ a Y . {\displaystyle \omega _{Y{\tilde {|}}X}=\omega _{X|Y}\;{\widetilde {\phi \,}}\;a_{Y}.}

Using these inverted conditionals together with the opinion ω X {\displaystyle \omega _{X}} subjective deduction denoted by the operator ⊚ {\displaystyle \circledcirc } can be used to abduce the marginal opinion ω Y ‖ ¯ X {\displaystyle \omega _{Y\,{\overline {\|}}\,X}}. The equality between the different expressions for subjective abduction is given below:

ω Y ‖ ~ X = ω X ∣ Y ⊚ ~ ω X = ( ω X ∣ Y ϕ ~ a Y ) ⊚ ω X = ω Y | ~ X ⊚ ω X . {\displaystyle {\begin{aligned}\omega _{Y\,{\widetilde {\|}}\,X}&=\omega _{X\mid Y}\;{\widetilde {\circledcirc }}\;\omega _{X}\\&=(\omega _{X\mid Y}\;{\widetilde {\phi \,}}\;a_{Y})\;\circledcirc \;\omega _{X}\\&=\omega _{Y{\widetilde {|}}X}\;\circledcirc \;\omega _{X}\;.\end{aligned}}}

The symbolic notation for subjective abduction is " ‖ ~ {\displaystyle {\widetilde {\|}}}", and the operator itself is denoted as " ⊚ ~ {\displaystyle {\widetilde {\circledcirc }}}". The operator for the subjective Bayes' theorem is denoted " ϕ ~ {\displaystyle {\widetilde {\phi \,}}}", and subjective deduction is denoted " ⊚ {\displaystyle \circledcirc }".[15]

The advantage of using subjective logic abduction compared to probabilistic abduction is that both aleatoric and epistemic uncertainty about the input argument probabilities can be explicitly expressed and taken into account during the analysis. It is thus possible to perform abductive analysis in the presence of uncertain arguments, which naturally results in degrees of uncertainty in the output conclusions.
History

The idea that the simplest, most easily verifiable solution should be preferred over its more complicated counterparts is a very old one. To this point, George Pólya, in his treatise on problem-solving, makes reference to the following Latin truism: simplex sigillum veri (simplicity is the seal of truth).[16]
[icon]	
This section needs expansion with: This deals only with Peirce and no other contributors or critics: other relevant histories should be added, and material that overlaps with the article on Peirce should be removed. You can help by adding missing information. (June 2020)
Introduction and development by Peirce
Overview

The American philosopher Charles Sanders Peirce introduced abduction into modern logic. Over the years he called such inference hypothesis, abduction, presumption, and retroduction. He considered it a topic in logic as a normative field in philosophy, not in purely formal or mathematical logic, and eventually as a topic also in economics of research.

As two stages of the development, extension, etc., of a hypothesis in scientific inquiry, abduction and also induction are often collapsed into one overarching concept—the hypothesis. That is why, in the scientific method known from Galileo and Bacon, the abductive stage of hypothesis formation is conceptualized simply as induction. Thus, in the twentieth century this collapse was reinforced by Karl Popper's explication of the hypothetico-deductive model, where the hypothesis is considered to be just "a guess"[17] (in the spirit of Peirce). However, when the formation of a hypothesis is considered the result of a process it becomes clear that this "guess" has already been tried and made more robust in thought as a necessary stage of its acquiring the status of hypothesis. Indeed, many abductions are rejected or heavily modified by subsequent abductions before they ever reach this stage.

Before 1900, Peirce treated abduction as the use of a known rule to explain an observation. For instance: it is a known rule that, if it rains, grass gets wet; so, to explain the fact that the grass on this lawn is wet, one abduces that it has rained. Abduction can lead to false conclusions if other rules that might explain the observation are not taken into account—e.g. the grass could be wet from dew. This remains the common use of the term "abduction" in the social sciences and in artificial intelligence.

Peirce consistently characterized it as the kind of inference that originates a hypothesis by concluding in an explanation, though an unassured one, for some very curious or surprising (anomalous) observation stated in a premise. As early as 1865 he wrote that all conceptions of cause and force are reached through hypothetical inference; in the 1900s he wrote that all explanatory content of theories is reached through abduction. In other respects Peirce revised his view of abduction over the years.[18]

In later years his view came to be:

    Abduction is guessing.[19] It is "very little hampered" by rules of logic.[20] Even a well-prepared mind's individual guesses are more frequently wrong than right.[21] But the success of our guesses far exceeds that of random luck and seems born of attunement to nature by instinct[22] (some speak of intuition in such contexts[23]).
    Abduction guesses a new or outside idea so as to account in a plausible, instinctive, economical way for a surprising or very complicated phenomenon. That is its proximate aim.[22]
    Its longer aim is to economize inquiry itself. Its rationale is inductive: it works often enough, is the only source of new ideas, and has no substitute in expediting the discovery of new truths.[24] Its rationale especially involves its role in coordination with other modes of inference in inquiry. It is inference to explanatory hypotheses for selection of those best worth trying.
    Pragmatism is the logic of abduction. Upon the generation of an explanation (which he came to regard as instinctively guided), the pragmatic maxim gives the necessary and sufficient logical rule to abduction in general. The hypothesis, being insecure, needs to have conceivable[25] implications for informed practice, so as to be testable[26][27] and, through its trials, to expedite and economize inquiry. The economy of research is what calls for abduction and governs its art.[28]

Writing in 1910, Peirce admits that "in almost everything I printed before the beginning of this century I more or less mixed up hypothesis and induction" and he traces the confusion of these two types of reasoning to logicians' too "narrow and formalistic a conception of inference, as necessarily having formulated judgments from its premises."[29]

He started out in the 1860s treating hypothetical inference in a number of ways which he eventually peeled away as inessential or, in some cases, mistaken:

    as inferring the occurrence of a character (a characteristic) from the observed combined occurrence of multiple characters which its occurrence would necessarily involve;[30] for example, if any occurrence of A is known to necessitate occurrence of B, C, D, E, then the observation of B, C, D, E suggests by way of explanation the occurrence of A. (But by 1878 he no longer regarded such multiplicity as common to all hypothetical inference.[31]Wikisource)
    as aiming for a more or less probable hypothesis (in 1867 and 1883 but not in 1878; anyway by 1900 the justification is not probability but the lack of alternatives to guessing and the fact that guessing is fruitful;[32] by 1903 he speaks of the "likely" in the sense of nearing the truth in an "indefinite sense";[33] by 1908 he discusses plausibility as instinctive appeal.[22]) In a paper dated by editors as circa 1901, he discusses "instinct" and "naturalness", along with the kind of considerations (low cost of testing, logical caution, breadth, and incomplexity) that he later calls methodeutical.[34]
    as induction from characters (but as early as 1900 he characterized abduction as guessing[32])
    as citing a known rule in a premise rather than hypothesizing a rule in the conclusion (but by 1903 he allowed either approach[20][35])
    as basically a transformation of a deductive categorical syllogism[31] (but in 1903 he offered a variation on modus ponens instead,[20] and by 1911 he was unconvinced that any one form covers all hypothetical inference[36]).

The Natural Classification of Arguments (1867)

In Peirce's On the Natural Classification of Arguments (1867),[30] hypothetical inference always deals with a cluster of characters (call them P′, P′′, P′′′, etc.) known to occur at least whenever a certain character (M) occurs. Note that categorical syllogisms have elements traditionally called middles, predicates, and subjects. For example: All men [middle] are mortal [predicate]; Socrates [subject] is a man [middle]; ergo Socrates [subject] is mortal [predicate]". Below, 'M' stands for a middle; 'P' for a predicate; 'S' for a subject. Peirce held that all deduction can be put into the form of the categorical syllogism Barbara (AAA-1).

    [Deduction]. 	Induction. 	Hypothesis.

    [Any] M is P
    [Any] S is M
    ∴ {\displaystyle \therefore } [Any] S is P.
    	

    S′, S′′, S′′′, &c. are taken at random as M's;
    S′, S′′, S′′′, &c. are P:
    ∴ {\displaystyle \therefore } Any M is probably P.
    	

    Any M is, for instance, P′, P′′, P′′′, &c.;
    S is P′, P′′, P′′′, &c.:
    ∴ {\displaystyle \therefore } S is probably M.

Deduction, Induction, and Hypothesis (1878)

In 1878, in Deduction, Induction, and Hypothesis,[31] there is no longer a need for multiple characters or predicates in order for an inference to be hypothetical, although it is still helpful. Moreover, Peirce no longer poses hypothetical inference as concluding in a probable hypothesis. In the forms themselves, it is understood but not explicit that induction involves random selection and that hypothetical inference involves response to a "very curious circumstance". The forms instead emphasize the modes of inference as rearrangements of one another's propositions (without the bracketed hints shown below).
Deduction. 	Induction. 	Hypothesis.

Rule: All the beans from this bag are white.
Case: These beans are from this bag.
∴ {\displaystyle \therefore } Result: These beans are white.
	

Case: These beans are [randomly selected] from this bag.
Result: These beans are white.
∴ {\displaystyle \therefore } Rule: All the beans from this bag are white.
	

Rule: All the beans from this bag are white.
Result: These beans [oddly] are white.
∴ {\displaystyle \therefore } Case: These beans are from this bag.
A Theory of Probable Inference (1883)

Peirce long treated abduction in terms of induction from characters or traits (weighed, not counted like objects), explicitly so in his influential 1883 "A theory of probable inference", in which he returns to involving probability in the hypothetical conclusion.[37] Like Deduction, Induction, and Hypothesis in 1878, it was widely read (see the historical books on statistics by Stephen Stigler), unlike his later amendments of his conception of abduction. Today abduction remains most commonly understood as induction from characters and extension of a known rule to cover unexplained circumstances.

Sherlock Holmes used this method of reasoning in the stories of Arthur Conan Doyle, although Holmes refers to it as "deductive reasoning".[38][39][40]
Minute Logic (1902) and after

In 1902 Peirce wrote that he now regarded the syllogistical forms and the doctrine of extension and comprehension (i.e., objects and characters as referenced by terms), as being less fundamental than he had earlier thought.[41] In 1903 he offered the following form for abduction:[20]

    The surprising fact, C, is observed;
    But if A were true, C would be a matter of course,
    Hence, there is reason to suspect that A is true.

The hypothesis is framed, but not asserted, in a premise, then asserted as rationally suspectable in the conclusion. Thus, as in the earlier categorical syllogistic form, the conclusion is formulated from some premise(s). But all the same the hypothesis consists more clearly than ever in a new or outside idea beyond what is known or observed. Induction in a sense goes beyond observations already reported in the premises, but it merely amplifies ideas already known to represent occurrences, or tests an idea supplied by hypothesis; either way it requires previous abductions in order to get such ideas in the first place. Induction seeks facts to test a hypothesis; abduction seeks a hypothesis to account for facts.

Note that the hypothesis ("A") could be of a rule. It need not even be a rule strictly necessitating the surprising observation ("C"), which needs to follow only as a "matter of course"; or the "course" itself could amount to some known rule, merely alluded to, and also not necessarily a rule of strict necessity. In the same year, Peirce wrote that reaching a hypothesis may involve placing a surprising observation under either a newly hypothesized rule or a hypothesized combination of a known rule with a peculiar state of facts, so that the phenomenon would be not surprising but instead either necessarily implied or at least likely.[35]

Peirce did not remain quite convinced about any such form as the categorical syllogistic form or the 1903 form. In 1911, he wrote, "I do not, at present, feel quite convinced that any logical form can be assigned that will cover all 'Retroductions'. For what I mean by a Retroduction is simply a conjecture which arises in the mind."[36]
Pragmatism

In 1901 Peirce wrote, "There would be no logic in imposing rules, and saying that they ought to be followed, until it is made out that the purpose of hypothesis requires them."[42] In 1903 Peirce called pragmatism "the logic of abduction" and said that the pragmatic maxim gives the necessary and sufficient logical rule to abduction in general.[27] The pragmatic maxim is:

    Consider what effects, that might conceivably have practical bearings, we conceive the object of our conception to have. Then, our conception of these effects is the whole of our conception of the object.

It is a method for fruitful clarification of conceptions by equating the meaning of a conception with the conceivable practical implications of its object's conceived effects. Peirce held that that is precisely tailored to abduction's purpose in inquiry, the forming of an idea that could conceivably shape informed conduct. In various writings in the 1900s[28][43] he said that the conduct of abduction (or retroduction) is governed by considerations of economy, belonging in particular to the economics of research. He regarded economics as a normative science whose analytic portion might be part of logical methodeutic (that is, theory of inquiry).[44]
Three levels of logic about abduction

Peirce came over the years to divide (philosophical) logic into three departments:

    Stechiology, or speculative grammar, on the conditions for meaningfulness. Classification of signs (semblances, symptoms, symbols, etc.) and their combinations (as well as their objects and interpretants).
    Logical critic, or logic proper, on validity or justifiability of inference, the conditions for true representation. Critique of arguments in their various modes (deduction, induction, abduction).
    Methodeutic, or speculative rhetoric, on the conditions for determination of interpretations. Methodology of inquiry in its interplay of modes.

Peirce had, from the start, seen the modes of inference as being coordinated together in scientific inquiry and, by the 1900s, held that hypothetical inference in particular is inadequately treated at the level of critique of arguments.[26][27] To increase the assurance of a hypothetical conclusion, one needs to deduce implications about evidence to be found, predictions which induction can test through observation so as to evaluate the hypothesis. That is Peirce's outline of the scientific method of inquiry, as covered in his inquiry methodology, which includes pragmatism or, as he later called it, pragmaticism, the clarification of ideas in terms of their conceivable implications regarding informed practice.
Classification of signs

As early as 1866,[45] Peirce held that:

    Hypothesis (abductive inference) is inference through an icon (also called a likeness).
    Induction is inference through an index (a sign by factual connection); a sample is an index of the totality from which it is drawn.
    Deduction is inference through a symbol (a sign by interpretive habit irrespective of resemblance or connection to its object).

In 1902, Peirce wrote that, in abduction: "It is recognized that the phenomena are like, i.e. constitute an Icon of, a replica of a general conception, or Symbol."[46]
Critique of arguments

At the critical level Peirce examined the forms of abductive arguments (as discussed above), and came to hold that the hypothesis should economize explanation for plausibility in terms of the feasible and natural. In 1908 Peirce described this plausibility in some detail.[22] It involves not likeliness based on observations (which is instead the inductive evaluation of a hypothesis), but instead optimal simplicity in the sense of the "facile and natural", as by Galileo's natural light of reason and as distinct from "logical simplicity" (Peirce does not dismiss logical simplicity entirely but sees it in a subordinate role; taken to its logical extreme it would favor adding no explanation to the observation at all). Even a well-prepared mind guesses oftener wrong than right, but our guesses succeed better than random luck at reaching the truth or at least advancing the inquiry, and that indicates to Peirce that they are based in instinctive attunement to nature, an affinity between the mind's processes and the processes of the real, which would account for why appealingly "natural" guesses are the ones that oftenest (or least seldom) succeed; to which Peirce added the argument that such guesses are to be preferred since, without "a natural bent like nature's", people would have no hope of understanding nature. In 1910 Peirce made a three-way distinction between probability, verisimilitude, and plausibility, and defined plausibility with a normative "ought": "By plausibility, I mean the degree to which a theory ought to recommend itself to our belief independently of any kind of evidence other than our instinct urging us to regard it favorably."[47] For Peirce, plausibility does not depend on observed frequencies or probabilities, or on verisimilitude, or even on testability, which is not a question of the critique of the hypothetical inference as an inference, but rather a question of the hypothesis's relation to the inquiry process.

The phrase "inference to the best explanation" (not used by Peirce but often applied to hypothetical inference) is not always understood as referring to the most simple and natural hypotheses (such as those with the fewest assumptions). However, in other senses of "best", such as "standing up best to tests", it is hard to know which is the best explanation to form, since one has not tested it yet. Still, for Peirce, any justification of an abductive inference as "good" is not completed upon its formation as an argument (unlike with induction and deduction) and instead depends also on its methodological role and promise (such as its testability) in advancing inquiry.[26][27][48]
Methodology of inquiry

At the methodeutical level Peirce held that a hypothesis is judged and selected[26] for testing because it offers, via its trial, to expedite and economize the inquiry process itself toward new truths, first of all by being testable and also by further economies,[28] in terms of cost, value, and relationships among guesses (hypotheses). Here, considerations such as probability, absent from the treatment of abduction at the critical level, come into play. For examples:

    Cost: A simple but low-odds guess, if low in cost to test for falsity, may belong first in line for testing, to get it out of the way. If surprisingly it stands up to tests, that is worth knowing early in the inquiry, which otherwise might have stayed long on a wrong though seemingly likelier track.
    Value: A guess is intrinsically worth testing if it has instinctual plausibility or reasoned objective probability, while subjective likelihood, though reasoned, can be treacherous.
    Interrelationships: Guesses can be chosen for trial strategically for their
        caution, for which Peirce gave as an example the game of Twenty Questions,
        breadth of applicability to explain various phenomena, and
        incomplexity, that of a hypothesis that seems too simple but whose trial "may give a good 'leave', as the billiard-players say", and be instructive for the pursuit of various and conflicting hypotheses that are less simple.[49]

Uberty

Peirce[50] indicated that abductive reasoning is driven by the need for "economy in research"—the expected fact-based productivity of hypotheses, prior to deductive and inductive processes of verification. A key concept proposed by him in this regard is "uberty"[51]—the expected fertility and pragmatic value of reasoning. This concept seems to be gaining support via association to the Free Energy Principle.[52]
Gilbert Harman (1965)

Gilbert Harman was a professor of philosophy at Princeton University. Harman's 1965 account of the role of "inference to the best explanation" – inferring the existence of that which we need for the best explanation of observable phenomena – has been very influential.
Stephen Jay Gould (1995)

Stephen Jay Gould, in answering the Omphalos hypothesis, claimed that only hypotheses that can be proved incorrect lie within the domain of science and only these hypotheses are good explanations of facts worth inferring to.[53]

    [W]hat is so desperately wrong with Omphalos? Only this really (and perhaps paradoxically): that we can devise no way to find out whether it is wrong—or for that matter, right. Omphalos is the classic example of an utterly untestable notion, for the world will look exactly the same in all its intricate detail whether fossils and strata are prochronic [signs of a fictitious past] or products of an extended history. ... Science is a procedure for testing and rejecting hypotheses, not a compendium of certain knowledge. Claims that can be proved incorrect lie within its domain. ... But theories that cannot be tested in principle are not part of science. ... [W]e reject Omphalos as useless, not wrong.

Applications
Artificial intelligence

Applications in artificial intelligence include fault diagnosis, belief revision, and automated planning. The most direct application of abduction is that of automatically detecting faults in systems: given a theory relating faults with their effects and a set of observed effects, abduction can be used to derive sets of faults that are likely to be the cause of the problem.[4]
Medicine

In medicine, abduction can be seen as a component of clinical evaluation and judgment.[54][55] The Internist-I diagnostic system, the first AI system that covered the field of Internal Medicine, used abductive reasoning to converge on the most likely causes of a set of patient symptoms that it acquired through an interactive dialog with an expert user.[56]
Automated planning

Abduction can also be used to model automated planning.[57] Given a logical theory relating action occurrences with their effects (for example, a formula of the event calculus), the problem of finding a plan for reaching a state can be modeled as the problem of abducting a set of literals implying that the final state is the goal state.
Intelligence analysis

In intelligence analysis, analysis of competing hypotheses and Bayesian networks, probabilistic abductive reasoning is used extensively. Similarly in medical diagnosis and legal reasoning, the same methods are being used, although there have been many examples of errors, especially caused by the base rate fallacy and the prosecutor's fallacy.
Belief revision
icon	
This section does not cite any sources. Please help improve this section by adding citations to reliable sources. Unsourced material may be challenged and removed. (January 2019) (Learn how and when to remove this message)

Belief revision, the process of adapting beliefs in view of new information, is another field in which abduction has been applied. The main problem of belief revision is that the new information may be inconsistent with the prior web of beliefs, while the result of the incorporation cannot be inconsistent. The process of updating the web of beliefs can be done by the use of abduction: once an explanation for the observation has been found, integrating it does not generate inconsistency.

In 1992 Peter Gärdenfors presented a paper[58] which contained a brief survey of the area of belief revision and its relation to updating of logical databases, and explores the relationship between belief revision and nonmonotonic logic.

This use of abduction is not straightforward, as adding propositional formulae to other propositional formulae can only make inconsistencies worse. Instead, abduction is done at the level of the ordering of preference of the possible worlds. Preference models use fuzzy logic or utility models.
Philosophy of science

In the philosophy of science, abduction has been the key inference method to support scientific realism, and much of the debate about scientific realism is focused on whether abduction is an acceptable method of inference.[59]
Historical linguistics

In historical linguistics, abduction during language acquisition is often taken to be an essential part of processes of language change such as reanalysis and analogy.[60]
Applied linguistics

In applied linguistics research, abductive reasoning is starting to be used as an alternative explanation to inductive reasoning, in recognition of anticipated outcomes of qualitative inquiry playing a role in shaping the direction of analysis. It is defined as "The use of an unclear premise based on observations, pursuing theories to try to explain it" (Rose et al., 2020, p. 258)[61][62]
Anthropology

In anthropology, Alfred Gell in his influential book Art and Agency defined abduction (after Eco[63]) as "a case of synthetic inference 'where we find some very curious circumstances, which would be explained by the supposition that it was a case of some general rule, and thereupon adopt that supposition'".[64] Gell criticizes existing "anthropological" studies of art for being too preoccupied with aesthetic value and not preoccupied enough with the central anthropological concern of uncovering "social relationships", specifically the social contexts in which artworks are produced, circulated, and received.[65] Abduction is used as the mechanism for getting from art to agency. That is, abduction can explain how works of art inspire a sensus communis: the commonly held views shared by members that characterize a given society.[66]

The question Gell asks in the book is, "how does it initially 'speak' to people?" He answers by saying that "No reasonable person could suppose that art-like relations between people and things do not involve at least some form of semiosis."[64] However, he rejects any intimation that semiosis can be thought of as a language because then he would have to admit to some pre-established existence of the sensus communis that he wants to claim only emerges afterwards out of art. Abduction is the answer to this conundrum because the tentative nature of the abduction concept (Peirce likened it to guessing) means that not only can it operate outside of any pre-existing framework, but moreover, it can actually intimate the existence of a framework. As Gell reasons in his analysis, the physical existence of the artwork prompts the viewer to perform an abduction that imbues the artwork with intentionality. A statue of a goddess, for example, in some senses actually becomes the goddess in the mind of the beholder; and represents not only the form of the deity but also her intentions (which are adduced from the feeling of her very presence). Therefore, through abduction, Gell claims that art can have the kind of agency that plants the seeds that grow into cultural myths. The power of agency is the power to motivate actions and inspire ultimately the shared understanding that characterizes any given society.[66]
Computer programming

In formal methods, logic is used to specify and prove properties of computer programs. Abduction has been used in mechanized reasoning tools to increase the level of automation of the proof activity.

A technique known as bi-abduction, which mixes abduction and the frame problem, was used to scale reasoning techniques for memory properties to millions of lines of code;[67] logic-based abduction was used to infer pre-conditions for individual functions in a program, relieving the human of the need to do so. It led to a program-proof startup company, which was acquired by Facebook,[68] and the Infer program analysis tool, which led to thousands of bugs being prevented in industrial codebases.[69]

In addition to inference of function preconditions, abduction has been used to automate inference of invariants for program loops,[70] inference of specifications of unknown code,[71] and in synthesis of the programs themselves.[72] 
----
Automated theorem proving (also known as ATP or automated deduction) is a subfield of automated reasoning and mathematical logic dealing with proving mathematical theorems by computer programs. Automated reasoning over mathematical proof was a major motivating factor for the development of computer science.
Logical foundations

While the roots of formalized logic go back to Aristotle, the end of the 19th and early 20th centuries saw the development of modern logic and formalized mathematics. Frege's Begriffsschrift (1879) introduced both a complete propositional calculus and what is essentially modern predicate logic.[1] His Foundations of Arithmetic, published in 1884,[2] expressed (parts of) mathematics in formal logic. This approach was continued by Russell and Whitehead in their influential Principia Mathematica, first published 1910–1913,[3] and with a revised second edition in 1927.[4] Russell and Whitehead thought they could derive all mathematical truth using axioms and inference rules of formal logic, in principle opening up the process to automation.[5] In 1920, Thoralf Skolem simplified a previous result by Leopold Löwenheim, leading to the Löwenheim–Skolem theorem and, in 1930, to the notion of a Herbrand universe and a Herbrand interpretation that allowed (un)satisfiability of first-order formulas (and hence the validity of a theorem) to be reduced to (potentially infinitely many) propositional satisfiability problems.[6]

In 1929, Mojżesz Presburger showed that the first-order theory of the natural numbers with addition and equality (now called Presburger arithmetic in his honor) is decidable and gave an algorithm that could determine if a given sentence in the language was true or false.[7][8]

However, shortly after this positive result, Kurt Gödel published On Formally Undecidable Propositions of Principia Mathematica and Related Systems (1931), showing that in any sufficiently strong axiomatic system, there are true statements that cannot be proved in the system.[9] This topic was further developed in the 1930s by Alonzo Church and Alan Turing, who on the one hand gave two independent but equivalent definitions of computability, and on the other gave concrete examples of undecidable questions.[10]
First implementations

In 1954, Martin Davis programmed Presburger's algorithm for a JOHNNIAC vacuum-tube computer at the Institute for Advanced Study in Princeton, New Jersey. According to Davis, "Its great triumph was to prove that the sum of two even numbers is even".[8][11] More ambitious was the Logic Theorist in 1956, a deduction system for the propositional logic of the Principia Mathematica, developed by Allen Newell, Herbert A. Simon and J. C. Shaw. Also running on a JOHNNIAC, the Logic Theorist constructed proofs from a small set of propositional axioms and three deduction rules: modus ponens, (propositional) variable substitution, and the replacement of formulas by their definition. The system used heuristic guidance, and managed to prove 38 of the first 52 theorems of the Principia.[8]

The "heuristic" approach of the Logic Theorist tried to emulate human mathematicians, and could not guarantee that a proof could be found for every valid theorem even in principle. In contrast, other, more systematic algorithms achieved, at least theoretically, completeness for first-order logic. Initial approaches relied on the results of Herbrand and Skolem to convert a first-order formula into successively larger sets of propositional formulae by instantiating variables with terms from the Herbrand universe. The propositional formulas could then be checked for unsatisfiability using a number of methods. Gilmore's program used conversion to disjunctive normal form, a form in which the satisfiability of a formula is obvious.[8][12]
Decidability of the problem

Depending on the underlying logic, the problem of deciding the validity of a formula varies from trivial to impossible. For the common case of propositional logic, the problem is decidable but co-NP-complete, and hence only exponential-time algorithms are believed to exist for general proof tasks.[13] For a first-order predicate calculus, Gödel's completeness theorem states that the theorems (provable statements) are exactly the semantically valid well-formed formulas, so the valid formulas are computably enumerable: given unbounded resources, any valid formula can eventually be proven. However, invalid formulas (those that are not entailed by a given theory), cannot always be recognized.[14]

The above applies to first-order theories, such as Peano arithmetic. However, for a specific model that may be described by a first-order theory, some statements may be true but undecidable in the theory used to describe the model. For example, by Gödel's incompleteness theorem, we know that any consistent theory whose axioms are true for the natural numbers cannot prove all first-order statements true for the natural numbers, even if the list of axioms is allowed to be infinite enumerable.[15] It follows that an automated theorem prover will fail to terminate while searching for a proof precisely when the statement being investigated is undecidable in the theory being used, even if it is true in the model of interest. Despite this theoretical limit, in practice, theorem provers can solve many hard problems, even in models that are not fully described by any first-order theory (such as the integers).
Related problems

A simpler, but related, problem is proof verification, where an existing proof for a theorem is certified valid. For this, it is generally required that each individual proof step can be verified by a primitive recursive function or program, and hence the problem is always decidable.

Since the proofs generated by automated theorem provers are typically very large, the problem of proof compression is crucial, and various techniques aiming at making the prover's output smaller, and consequently more easily understandable and checkable, have been developed.

Proof assistants require a human user to give hints to the system. Depending on the degree of automation, the prover can essentially be reduced to a proof checker, with the user providing the proof in a formal way, or significant proof tasks can be performed automatically. Interactive provers are used for a variety of tasks, but even fully automatic systems have proved a number of interesting and hard theorems, including at least one that has eluded human mathematicians for a long time, namely the Robbins conjecture.[16][17] However, these successes are sporadic, and work on hard problems usually requires a proficient user.

Another distinction is sometimes drawn between theorem proving and other techniques, where a process is considered to be theorem proving if it consists of a traditional proof, starting with axioms and producing new inference steps using rules of inference. Other techniques would include model checking, which, in the simplest case, involves brute-force enumeration of many possible states (although the actual implementation of model checkers requires much cleverness, and does not simply reduce to brute force).

There are hybrid theorem proving systems that use model checking as an inference rule. There are also programs that were written to prove a particular theorem, with a (usually informal) proof that if the program finishes with a certain result, then the theorem is true. A good example of this was the machine-aided proof of the four color theorem, which was very controversial as the first claimed mathematical proof that was essentially impossible to verify by humans due to the enormous size of the program's calculation (such proofs are called non-surveyable proofs). Another example of a program-assisted proof is the one that shows that the game of Connect Four can always be won by the first player.
Applications

Commercial use of automated theorem proving is mostly concentrated in integrated circuit design and verification. Since the Pentium FDIV bug, the complicated floating point units of modern microprocessors have been designed with extra scrutiny. AMD, Intel and others use automated theorem proving to verify that division and other operations are correctly implemented in their processors.[18]

Other uses of theorem provers include program synthesis, constructing programs that satisfy a formal specification.[19] Automated theorem provers have been integrated with proof assistants, including Isabelle/HOL.[20]

Applications of theorem provers are also found in natural language processing and formal semantics, where they are used to analyze discourse representations.[21][22]
First-order theorem proving

In the late 1960s agencies funding research in automated deduction began to emphasize the need for practical applications.[citation needed] One of the first fruitful areas was that of program verification whereby first-order theorem provers were applied to the problem of verifying the correctness of computer programs in languages such as Pascal, Ada, etc. Notable among early program verification systems was the Stanford Pascal Verifier developed by David Luckham at Stanford University.[23][24][25] This was based on the Stanford Resolution Prover also developed at Stanford using John Alan Robinson's resolution principle. This was the first automated deduction system to demonstrate an ability to solve mathematical problems that were announced in the Notices of the American Mathematical Society before solutions were formally published.[citation needed]

First-order theorem proving is one of the most mature subfields of automated theorem proving. The logic is expressive enough to allow the specification of arbitrary problems, often in a reasonably natural and intuitive way. On the other hand, it is still semi-decidable, and a number of sound and complete calculi have been developed, enabling fully automated systems.[26] More expressive logics, such as higher-order logics, allow the convenient expression of a wider range of problems than first-order logic, but theorem proving for these logics is less well developed.[27][28]
Relationship with SMT

There is substantial overlap between first-order automated theorem provers and SMT solvers. Generally, automated theorem provers focus on supporting full first-order logic with quantifiers, whereas SMT solvers focus more on supporting various theories (interpreted predicate symbols). ATPs excel at problems with lots of quantifiers, whereas SMT solvers do well on large problems without quantifiers.[29] The line is blurry enough that some ATPs participate in SMT-COMP, while some SMT solvers participate in CASC.[30]
Benchmarks, competitions, and sources

The quality of implemented systems has benefited from the existence of a large library of standard benchmark examples—the Thousands of Problems for Theorem Provers (TPTP) Problem Library[31]—as well as from the CADE ATP System Competition (CASC), a yearly competition of first-order systems for many important classes of first-order problems.

Some important systems (all have won at least one CASC competition division) are listed below.

    E is a high-performance prover for full first-order logic, but built on a purely equational calculus, originally developed in the automated reasoning group of Technical University of Munich under the direction of Wolfgang Bibel, and now at Baden-Württemberg Cooperative State University in Stuttgart.
    Otter, developed at the Argonne National Laboratory, is based on first-order resolution and paramodulation. Otter has since been replaced by Prover9, which is paired with Mace4.
    SETHEO is a high-performance system based on the goal-directed model elimination calculus, originally developed by a team under direction of Wolfgang Bibel. E and SETHEO have been combined (with other systems) in the composite theorem prover E-SETHEO.
    Vampire was originally developed and implemented at Manchester University by Andrei Voronkov and Kryštof Hoder. It is now developed by a growing international team. It has won the FOF division (among other divisions) at the CADE ATP System Competition regularly since 2001.[32]
    Waldmeister is a specialized system for unit-equational first-order logic developed by Arnim Buch and Thomas Hillenbrand. It won the CASC UEQ division for fourteen consecutive years (1997–2010).
    SPASS is a first-order logic theorem prover with equality. This is developed by the research group Automation of Logic, Max Planck Institute for Computer Science.

The Theorem Prover Museum[33] is an initiative to conserve the sources of theorem prover systems for future analysis, since they are important cultural/scientific artefacts. It has the sources of many of the systems mentioned above.
Popular techniques
	
This article is in list format but may read better as prose. You can help by converting this article, if appropriate. Editing help is available. (December 2023)

    First-order resolution with unification
    Model elimination
    Method of analytic tableaux
    Superposition and term rewriting
    Model checking
    Mathematical induction[34]
    Binary decision diagrams
    DPLL
    Higher-order unification
    Quantifier elimination[35]

Software systems
See also: Proof assistant § Comparison, and Category:Theorem proving software systems
Comparison
					
					
					
					
					
					
					
					
					
					
					
					
					
					
					
Free software

    Alt-Ergo
    Automath
    CVC
    E
    IsaPlanner
    LCF
    Mizar
    NuPRL
    Paradox
    Prover9
    PVS
    SPARK (programming language)
    Twelf
    Z3 Theorem Prover
----
Now you understand why we must set the internat swarm to be able to be spawned n times within hardware constrains, starting with the base we have
