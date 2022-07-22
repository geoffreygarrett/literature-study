# Machine Learning Fundamentals

This section discusses the fundamentals of [ML]{acronym-label="ML"
acronym-form="singular+short"}, which underpin all
[ML]{acronym-label="ML" acronym-form="singular+short"} algorithms. In
this first part of our discussion (), we look at what it means for an
algorithm to \"learn\" and the kinds of experiences used in learning
that form the taxonomy of machine learning. The types of datasets
encountered in these learning algorithms are then briefly covered to
provide insight into the potential applications that are not covered in
this review. This is followed by distinguishing between the goal of
fitting training data and finding patterns that generalise to new data.
Finally, a ubiquitous concept in machine learning is covered:
*hyperparameters*, which are *settings* of a learning algorithm which
must be determined outside the learning algorithm itself.

This section provides a brief overview of the fundamentals of
[ML]{acronym-label="ML" acronym-form="singular+short"}, which apply to
all [ML]{acronym-label="ML" acronym-form="singular+short"} algorithms.
To begin, we must first comprehend what it means for an algorithm to
\"learn.\" The sorts of data these algorithms work with are then
evaluated to provide more information on the possible applications that
aren't directly addressed here. Finally, one of the most fundamental
concepts in machine learning is outlined: hyperparameters, which are
settings for a learning algorithm that must be determined outside of the
learning algorithm itself.

## Learning algorithms[\[ssec:learning_algorithms\]]{#ssec:learning_algorithms label="ssec:learning_algorithms"}

Generally speaking, a machine learning algorithm is a procedure for
learning from data. However correct this definition is, it provides
little insight into the relevant concepts in the field. A more succinct
definition is provided by Carnegie Mellon University professor Tom
Mitchell [@Mitchell97LearningAlgorithm]:

::: {.quote}
A computer program is said to learn from experience $E$ with respect to
some class of tasks $T$ and a performance measure $P$, if its
performance at tasks in $T$, as measured by $P$, improves with
experience $E$.
:::

This definition introduces the entities present during all machine
learning tasks. The entities will not be formally defined in the
following sections as it is far outside the scope of this literature and
is philosophical in nature. This section will instead cover examples of
each, which will provide practical insight on which the reader can build
their knowledge.

### The Task, $T$

There exist a plethora of tasks to which humanity has applied learning
algorithms to during the timeline of the field. It is common for
[ML]{acronym-label="ML" acronym-form="singular+short"} practitioners to
originate from domains outside that of computer science in order to
assess the feasibility of existing algorithms in their domain. There is,
however, a question that often presents itself to specialists in their
respective fields when they consider applying [ML]{acronym-label="ML"
acronym-form="singular+short"} to the existing problems in their field.
Why should a specialist opt for solving problems with
[ML]{acronym-label="ML" acronym-form="singular+short"} that have been
solved using tried and true techniques in their domain of expertise?
Goodfellow et al. [@Goodfellow-et-al-2016] provide an insightful
response to this:

::: {.quote}
Machine learning enables us to tackle tasks that are too difficult to
solve with fixed programs written and designed by human beings. From a
scientific and philosophical point of view, machine learning is
interesting because developing our understanding of it entails
developing our understanding of the principles that underlie
intelligence.
:::

An example of a field which has undergone dramatic changes in a short
period of time, with the advent of [DL]{acronym-label="DL"
acronym-form="singular+short"} and modern hardware, is
[CV]{acronym-label="CV" acronym-form="singular+short"}. Mahony et al.
discuss this in their paper with a focus on comparing
[DL]{acronym-label="DL" acronym-form="singular+short"} and
[CV]{acronym-label="CV" acronym-form="singular+short"}
[@Mahony-et-al-2020]. Their paper concludes that many
[CV]{acronym-label="CV" acronym-form="singular+short"} techniques
invented in the 20 years preceding the paper have become irrelevant as a
result of [DL]{acronym-label="DL" acronym-form="singular+short"}.
However, they emphasise the importance of the knowledge established in
those 20 years, arguing that *"knowledge is never obsolete\"*, as it
provides specialists with more tools and intuition when addressing
problems. Some typical applications of [CV]{acronym-label="CV"
acronym-form="singular+short"} are detailed, and although these may be
outperformed by [DL]{acronym-label="DL" acronym-form="singular+short"},
relying on [DL]{acronym-label="DL" acronym-form="singular+short"} in
some cases is overkill. They also point out some hybrid approaches
between [DL]{acronym-label="DL" acronym-form="singular+short"} and
[CV]{acronym-label="CV" acronym-form="singular+short"} which synergise,
saying that [CV]{acronym-label="CV" acronym-form="singular+short"}
provides improved performance in [DL]{acronym-label="DL"
acronym-form="singular+short"} by reducing training time. This
emphasises that specialists should not expect an end-all solution from
[ML]{acronym-label="ML" acronym-form="singular+short"} in addressing
their domain-specific problems, but rather as mentioned by Goodfellow et
al., strive for a better understanding of the principles that underlie
intelligence and, by extension, those that underlie the practitioner's
domain-specific problems.

-   **Classification**: In this task, the learning algorithm is expected
    to produce a function $f: \mathbb{R^n}\rightarrow \{1,...,k\}$. When
    $y=f(\mathbf{x})$, the model assigns a provided input, $\mathbf{x}$
    to a category identified by numeric code $y$. An example of this
    would be the mapping of a grayscale image,
    $\mathbf{x}\in\gls{set:R}^2$ to a value corresponding to a numerical
    encoding $f:\gls{set:R}^n\rightarrow\{\textrm{Cat},\;\textrm{Dog}\}$.

-   **Classification with missing inputs**: Classification becomes more
    challenging when the model input measurements are not always
    guaranteed to be the same. In this situation, the algorithm must
    learn the set of all function mappings arising from the possible
    combinations of input vectors that arise from missing subsets of
    inputs in $\mathbf{x}$. An example of this would be the
    classification of a medical diagnosis, as depending on the
    invasiveness of certain procedures, different subsets of
    measurements are available.

-   **Regression**: In this task, the learning algorithm is expected to
    predict a continuous numerical value for a given input. This is done
    by learning a function $f: \gls{set:R}^n\rightarrow\gls{set:R}$. the
    formulation is similar to classification, except for the output
    format. An example of this would be learning a function to predict
    the expected returns for a given investment given the state of the
    market, as is common in algorithmic trading.

-   **Transcription**: This type of task involves the learning algorithm
    observing a relatively unstructured input and transcribing it into
    some discrete textual form. An example is the transcription of an
    audio waveform containing speech into text.

-   **Machine translation**: In machine translation, the already
    structured input is mapped into a different language. This is common
    in the field of [NLP]{acronym-label="NLP"
    acronym-form="singular+short"}, where languages are translated, for
    example, English to and from French. This, however, is not limited
    to natural languages but can also be applied to programming
    languages.

-   **Structured output**: This task entails those where the output is a
    vector or a data structure which details important relationships
    between the contained elements. This task subsumes the prior two of
    transcription and machine translation. An example of this would be
    the parsing of the grammatical structure of a natural language
    sentence, addressed in [NLP]{acronym-label="NLP"
    acronym-form="singular+short"} and demonstrated by Collobert
    [@pmlr-v15-collobert11a].

-   **Anomaly detection**: In this task, a learning algorithm learns to
    sift through a set of events and classify events that are anomalous.
    These algorithms have been applied to credit card companies in the
    detection of fraud [@Tiwari2021] and safety in the aviation industry
    [@Janakiraman2016; @Basora2019].

-   **Synthesis and sampling**: In this task, learning algorithms
    generate **new** examples which appear to be drawn from the same
    underlying distribution but are not present in the training data. A
    recent example around which research is being done is the
    [GPT-3]{acronym-label="GPT-3" acronym-form="singular+short"} which
    can produce human-like text, given prompts as input [@Brown2020].

-   **Imputation of missing values**: In this task, the learning
    algorithm is given some input $x\in\gls{set:R}$ with some elements
    $x_i$ missing. The algorithm must then make predictions for the
    missing values. Emmanuel et al. discuss the latest methods in
    [ML]{acronym-label="ML" acronym-form="singular+short"} that address
    this task [@Emmanuel2021]. A recent example of the imputation of
    missing frames to add frames to videos for improved video smoothness
    is [RIFE]{acronym-label="RIFE" acronym-form="singular+short"}
    [@huang2020rife].

-   **Denoising**: In this learning task, the machine learning algorithm
    learns to generate a clean example $\mathbf{x}\in\gls{set:R}^n$ from
    a corrupted sample $\tilde{\mathbf{x}}\in\gls{set:R}^n$. Generally
    speaking the learner is predicting the probability distribution
    $p(\mathbf{x}|\tilde{\mathbf{x}})$.

-   **Density estimation** or **probability mass function estimation**
    involves the task of learning a function
    $p_\text{model}:\gls{set:R}^n\rightarrow{}\gls{set:R}$, where
    $p_\text{model}(\mathbf{x})$ is interpreted as a probability
    distribution function from which $\mathbf{x}$ was drawn. Effectively
    the algorithm learns the structure of the data that it has been
    provided with. Most of the aforementioned tasks involve learning
    this distribution function, at least implicitly. For example for the
    missing value imputation task, if $x_i$ is missing, and all other
    values $x_{-i}$ are given, then the algorithm learns the
    distribution $p(x_i|\mathbf{x}_{-i})$ which involves $p(\mathbf{x})$
    implicitly.

### The Performance, $P$[\[sec:ML-performance\]]{#sec:ML-performance label="sec:ML-performance"}

In order to assess the performance of our learning algorithm in task
$T$, we must have some quantitative measure of performance $P$ with
which we steer the algorithm towards the desired behaviour. $T$ and $P$
in this way are coupled, and must $P$ must be chosen according to $T$.
In this way, it can be unclear to the [ML]{acronym-label="ML"
acronym-form="singular+short"} practitioner what $P$ should be chosen.

Firstly, it is worth considering the true goal of the learning
algorithm. We wish for it to train according to some dataset such that
it can generalise to inputs that it has never seen before. After all,
there is little worth in a classification algorithm recognising cats and
dogs with 90% accuracy on its training dataset, if it performs with 20%
accuracy on an unseen **test** dataset. This previously mentioned
scenario is known as *overfitting* and is discussed in its own section
later in the chapter. For this reason, the first requirement for
managing our dataset is presented; there must be some **train-test
split**. This is further complicated by the **validation** dataset,
which is discussed later in the chapter.

Given that the input-output pair is $\{\gls{x_in_i}, \glsvec{fn:out}^{\gls{ml:ith:xy}}\}$,
[y_true_i]{acronym-label="y_true_i" acronym-form="singular+short"} is
commonly referred to as the *ground truth*, and
$\hat{f}(\gls{x_in_i})=\gls{y_pred_i}$ are referred to as the
*predictor* and *predicted output* respectively. The performance in the
form of a loss function [J]{acronym-label="J"
acronym-form="singular+short"} (also referred to as an objective or cost
function in literature) is then a function of
[y_true_i]{acronym-label="y_true_i" acronym-form="singular+short"} and
[y_pred_i]{acronym-label="y_pred_i" acronym-form="singular+short"},
therefore $\gls{J}(\gls{y_pred_i}, \glsvec{fn:out}^{\gls{ml:ith:xy}})$.

[ML]{acronym-label="ML" acronym-form="singular+short"} often borrows
from *approximation theory* vector norm notation
($\ell(\cdot)_p=||\cdot||_p$) to simplify notation in performance
metrics. See for the expressions of vector norms.

**Regression loss functions**

-   **[MSE]{acronym-label="MSE" acronym-form="singular+short"}** is the
    average of the squared differences between predictions and the
    ground truth. It is only concerned with the average magnitude of
    errors irrespective of their direction. Due to the squaring,
    predictions which have greater errors are more heavily penalized. A
    beneficial mathematical property of [MSE]{acronym-label="MSE"
    acronym-form="singular+short"} is that gradients can be easily
    calculated.
    $$\gls{J}_\text{\gls{MSE}} = \frac{1}{\gls{ml:m}}\sum_{i=1}^{\gls{ml:m}}||\gls{y_pred_i}-\glsvec{fn:out}^{\gls{ml:ith:xy}}||^2_2
            \label{eq:MSE}$$ $$\begin{aligned}
                \textrm{where, }
                \eqdesc{J}\text{,} \\
                \eqdesc{ml:m}\text{,} \\
                \eqdesc{y_pred_i}\text{,} \\
                \eqdesc{y_true_i}\text{.} \\
            \end{aligned}$$

-   **L2 error** is the average of the differences between predictions
    and the ground truth. It is the same as [MSE]{acronym-label="MSE"
    acronym-form="singular+short"} but without the squaring. Like
    [MSE]{acronym-label="MSE" acronym-form="singular+short"}, the
    gradient can be easily calculated.
    $$\gls{J}_\text{L2E} = \frac{1}{\gls{ml:m}}\sum_{i=1}^{\gls{ml:m}}||\gls{y_pred_i}-\glsvec{fn:out}^{\gls{ml:ith:xy}}||_2$$

-   **[MAE]{acronym-label="MAE" acronym-form="singular+short"}** or **L1
    loss** is the average of the sum of absolute differences between
    predictions and the ground truth. This measure is similar to
    [MSE]{acronym-label="MSE" acronym-form="singular+short"} in that it
    considers the magnitude of the errors and ignores their direction.
    However, [MAE]{acronym-label="MAE" acronym-form="singular+short"} is
    more difficult to compute gradients for, requiring linear
    programming. [MAE]{acronym-label="MAE"
    acronym-form="singular+short"} is also more resilient to large error
    values as a result of outliers, as it does not make use of a square.
    $$\gls{J}_\text{MAE} = \frac{1}{\gls{ml:m}}\sum_{i=1}^{\gls{ml:m}}||\gls{y_pred_i}-\glsvec{fn:out}^{\gls{ml:ith:xy}}||_1
            \label{eq:MAE}$$

-   **[MBE]{acronym-label="MBE" acronym-form="singular+short"}** is the
    sum of all differences between predictions and the ground truth.
    [MBE]{acronym-label="MBE" acronym-form="singular+short"} is not
    often used as a measure of the model error as high individual errors
    can produce a low [MBE]{acronym-label="MBE"
    acronym-form="singular+short"}. This metric is primarily used to
    measure the average bias of a prediction.
    $$\gls{J}_\text{MBE} = \frac{1}{\gls{ml:m}}\sum_{i=1}^{\gls{ml:m}}\sum_{j=1}^{\gls{np:dim}[_\gls{fn:out}]}(\gls{y_pred_ij}-\glsvec{fn:out}^{\gls{ml:ith:xy}}_\gls{ix:sub})$$
    $$\begin{aligned}
                \textrm{where, }
                \eqdesc{ml:n_y}\text{,} \\
                \eqdesc{y_pred_ij}\text{,} \\
                \eqdesc{y_true_ij}\text{.} \\
            \end{aligned}$$

-   **[MSLE]{acronym-label="MSLE" acronym-form="singular+short"}** can
    be interpreted as a measure of the ratio between predictions and the
    ground truth. This metric is used when it is desirable to achieve
    some relative measure of accuracy as opposed to an absolute. In
    other words, large and small discrepancies between the predicted and
    ground truth are dealt in a relative manner, resulting in similar
    loss value magnitudes. $$\begin{aligned}
                \gls{J}_\text{MSLE} &= \frac{1}{\gls{ml:m}}\sum_{i=1}^{\gls{ml:m}}\sum_{j=1}^{\gls{np:dim}[_\gls{fn:out}]}(\log{(\glsvec{fn:out}^{\gls{ml:ith:xy}}_\gls{ix:sub}+1)} - \log{(\gls{y_pred_ij}+1)})^2 \\
                &= \frac{1}{\gls{ml:m}}\sum_{i=1}^{\gls{ml:m}}\sum_{j=1}^{\gls{np:dim}[_\gls{fn:out}]}2\log(\frac{\glsvec{fn:out}^{\gls{ml:ith:xy}}_\gls{ix:sub}+1}{\gls{y_pred_ij}+1})
            \end{aligned}$$

-   **[CP]{acronym-label="CP" acronym-form="singular+short"}** or
    **[CS]{acronym-label="CS" acronym-form="singular+short"}** is a
    measure of similarity between two non-zero vectors of an inner
    product space. The metric is defined as the cosine of the angle
    between the two vectors. The metric is bounded in the interval
    $[-1,1]$, -1 for opposing orientation and 1 for the same
    orientation. This is not often useful in some
    [ML]{acronym-label="ML" acronym-form="singular+short"} tasks, so an
    alternative form known as the **[CD]{acronym-label="CD"
    acronym-form="singular+short"}** is used instead. The relation
    between the metrics is $\text{\gls{CD}}=1-\text{\gls{CP}}$, bounding
    the error in the interval $[0,2]$. $$\gls{J}_\text{\gls{CD}} =
            \frac{1}{\gls{ml:m}}\sum_{i=1}^{\gls{ml:m}}\bigg(1-\frac{\glsvec{fn:out}^{\gls{ml:ith:xy}}\cdot{}\gls{y_pred_i}}{||\glsvec{fn:out}^{\gls{ml:ith:xy}}||_2\cdot{}||\gls{y_pred_i}||_2}\bigg)$$

**Binary classification loss functions**\
Binary classification differs from multi-class in that the output is a
single value ([ml:n_y]{acronym-label="ml:n_y"
acronym-form="singular+short"}$=1$). For multi-class, the dimension of
the output is equal to the number of classes.

-   **[BCE]{acronym-label="BCE" acronym-form="singular+short"}**, also
    known as **maximum likelihood**, is a measure defined simply by the
    negative log-likelihood between the training data and model
    distribution. In the case where $\gls{y_true_i1}=1$ and
    $\gls{y_pred_i1}=0$, the log probability of the predicted output is
    undefined. For this reason, practitioners often add some small
    constant on the probability of [y_pred_i1]{acronym-label="y_pred_i1"
    acronym-form="singular+short"}, e.g. $10^{-5}$, resulting in
    [J]{acronym-label="J" acronym-form="singular+short"}$\approx{}5$.
    $$\gls{J}_\text{\gls{BCE}}=-\sum_{i=1}^{\gls{ml:m}}\bigg(\gls{y_true_i1}\log(\gls{y_pred_i1})+(1-\gls{y_true_i1})\log(1-\gls{y_pred_i1})\bigg)\label{eq:equation}$$

-   **[HL]{acronym-label="HL" acronym-form="singular+short"}** is
    measure which is often used for training binary
    [SVM]{acronym-label="SVM" acronym-form="singular+short"} classifiers
    [@Wu2007; @Liu2007; @Zhang2008]. The loss function assumes that
    [y_true_ij]{acronym-label="y_true_ij" acronym-form="singular+short"}
    and [y_pred_ij]{acronym-label="y_pred_ij"
    acronym-form="singular+short"} are in the bounds $[-1,1]$ for the
    classification task.
    $$\gls{J}_\text{\gls{HL}}=\sum_{i=1}^{\gls{ml:m}}\max(0, {1-\gls{y_true_i1}\cdot{}\gls{y_pred_i1}})$$

-   **[SHL]{acronym-label="SHL" acronym-form="singular+short"}**
    addresses the constant gradient present in [HL]{acronym-label="HL"
    acronym-form="singular+short"}, smoothing the loss function across
    the domain of the loss function
    ($\glsvec{fn:out}^{\gls{ml:ith:xy}}_\gls{ix:sub}\cdot{}\gls{y_pred_ij}$). This provides better
    gradient learning properties, as predictions that are more incorrect
    than others have a heavier associated gradient.
    $$\gls{J}_\text{\gls{SHL}}=\sum_{i=1}^{\gls{ml:m}}\max(0, {1-\gls{y_true_i1}\cdot{}\gls{y_pred_i1}})^2$$

**Multi-class classification loss functions** Multi-class classification
is formulated such that each possible class ($n_c$) is defined by an
element of the output vector. The loss functions here are similar to the
binary classification loss functions but adapted for $\gls{np:dim}[_\gls{fn:out}]=n_c$.

-   **[MCE]{acronym-label="MCE" acronym-form="singular+short"}**:
    $$\gls{J}_\text{\gls{MCE}}=-\sum_{i=1}^{\gls{ml:m}}\sum_{j=1}^{\gls{np:dim}[_\gls{fn:out}]}\glsvec{fn:out}^{\gls{ml:ith:xy}}_\gls{ix:sub}\log(\gls{y_pred_ij})$$

-   **[MHL]{acronym-label="MHL" acronym-form="singular+short"}**:
    $$\gls{J}_\text{\gls{MHL}}=\sum_{i=1}^{\gls{ml:m}}\sum_{j=1}^{\gls{np:dim}[_\gls{fn:out}]}\max(0, {1-\glsvec{fn:out}^{\gls{ml:ith:xy}}_\gls{ix:sub}\cdot{}\gls{y_pred_ij}})$$

-   **[MSHL]{acronym-label="MSHL" acronym-form="singular+short"}**:
    $$\gls{J}_\text{\gls{MSHL}}=\sum_{i=1}^{\gls{ml:m}}\sum_{j=1}^{\gls{np:dim}[_\gls{fn:out}]}\max(0, {1-\glsvec{fn:out}^{\gls{ml:ith:xy}}_\gls{ix:sub}\cdot{}\gls{y_pred_ij}})^2$$

### The Experience, $E$[\[sec:ML-experience\]]{#sec:ML-experience label="sec:ML-experience"}

Tasks provided to machine learning algorithms have different
formulations based on the experience made available. Learning algorithms
can be broadly categorised by the kind of experience that the algorithm
has access to during learning, from which it is generally expected to
learn some pattern of interest.

-   **Supervised learning** algorithms experience a dataset containing
    features (input) and labels (expected output) and are expected to
    learn a function mapping between the two. These algorithms can be
    further divided into two categories. **Classification algorithms**
    learn a mapping from an input to a discrete class label output.
    **Regression algorithms** learn a mapping from an input to a
    continuous value output.

-   **Unsupervised learning** algorithms experience a dataset containing
    only features and learn some useful properties of the structure of
    the dataset. This form of learning often addresses recognition
    problems in *association* & *clustering* [@barlow1999ul].

-   **Semi-supervised** is a middle ground between supervised learning
    (in which all training data is labelled) and unsupervised learning
    (in which no label data is provided) [@books/mit/06/CSZ2006]. Some
    example applications of this paradigm are dimensionality reduction
     [@Zhang2007], clustering [@Bair2013], and anomaly detection
    [@DBLP:journals/corr/abs-1805-06725].

-   **Reinforcement learning** are learning algorithms which rely
    exclusively on a series of reinforcements from their environment.
    These reinforcements can be positive (rewards) or negative
    (punishments). This category is discussed further in .

## Capacity, Overfitting and Underfitting[\[sec:capoverunder\]]{#sec:capoverunder label="sec:capoverunder"}

The primary objective in [ML]{acronym-label="ML"
acronym-form="singular+short"} is to find a model which performs well on
previously *unseen data*, which has not been used during the training or
fitting of a model. This ability of the learning algorithm is referred
to as its ability to *generalise*.

Typically the learning algorithm is trained according to a set of
*training data* with which we aim at reducing the learners **training
error**. Due to this data having been directly used to optimise the
model during training, we cannot expect this error to be representative
of the model's **generalisation error**. In order to derive an estimate
of the generalisation error, we separate part of the initial data set
into training and **test data**, from which the **test error** is
derived. Given some assumptions about the **data-generating process**,
such as the data being **[iid]{acronym-label="iid"
acronym-form="singular+short"}**, it can be said that the training error
is equal to the expectation of the test error.

An important concept in [ML]{acronym-label="ML"
acronym-form="singular+short"} is the *capacity* of the model. This is
defined informally as *"a model's \[\...\] ability to fit a wide variety
of functions\"* [@Goodfellow-et-al-2016 p. 111-112]. Machine learning
algorithms will generally perform their best when their capacity is
appropriate for the true complexity of the task they are expected to
perform, given the available training data. Models with insufficient
capacity will fail to perform complex tasks, and those with excess
capacity will perform complex tasks, but they may overfit the training
data.

One technique of controlling the capacity of an algorithm is to control
its **hypothesis space**, the set of possible functions that the
algorithm may use to solve the task at hand. In the case of a
polynomial, such as the arbitrary one seen in , this effect can be
intuitively related to the model's propensity to underfit and overfit a
set of training data.

[\[fig:underfitting\]]{#fig:underfitting label="fig:underfitting"}

[\[fig:appropriate-capacity\]]{#fig:appropriate-capacity
label="fig:appropriate-capacity"}

[\[fig:overfitting\]]{#fig:overfitting label="fig:overfitting"}

### No Free Lunch Theorem

The [NFLT]{acronym-label="NFLT" acronym-form="singular+short"},
sometimes abbreviated as [NFL]{acronym-label="NFL"
acronym-form="singular+short"}, is a simple but important concept in
[ML]{acronym-label="ML" acronym-form="singular+short"} and
optimisation [@Wolpert1997]. The theorem suggests that an optimisation
technique will perform equally well as any other when averaging its
performance over the set of all possible problems. This implies that
there is no single best technique for addressing an arbitrary problem.
Luke states the following in *essentials of
metaheuristics* [@luke2012essentials]:

::: {.quote}
The [NFL]{acronym-label="NFL" acronym-form="singular+short"} stated that
within certain constraints, over the space of all possible problems,
every optimisation technique will perform as well as every other one on
average (including Random Search)
:::

This argues that without having substantive information about the
fundamentals of the problem being modelled, choosing to apply a single
technique to an arbitrary problem will not yield a predictably better or
worse result than applying another to the same problem. Therefore, in
the case where the underlying process being optimised is not
well-understood, a variety of techniques should be applied.

However, in practice, knowledge to some degree is known about the
problem which is being optimised or to which a learning algorithm is
being applied. This theorem highlights the importance of having a clear
understanding of the problem at hand before applying a learning
algorithm or an optimisation technique. Domingos states  [@Domingos15]:

::: {.quote}
In the meantime, the practical consequence of the "no free lunch"
theorem is that there's no such thing as learning without knowledge.
Data alone is not enough.
:::

This theorem, in effect, motivates the true goal of machine learning, as
worded by Goodfellow et al. [@Goodfellow-et-al-2016 p. 116]:

::: {.quote}
This means that the goal of machine learning research is not to seek a
universal learning algorithm or the absolute best learning algorithm.
Instead, our goal is to understand what kinds of distributions are
relevant to the "real world" that an AI agent experiences, and what
kinds of machine learning algorithms perform well on data drawn from the
kinds of data-generating distributions we care about.
:::

This is an important insight and directly relates to a learning
algorithm's tendency to overfit or underfit. If we do not have a clear
understanding of the level of complexity of the task we are trying to
find a solution to with our learning algorithm, then we are likely to
overfit or underfit. If one assumes the complexity of the problem to be
higher than its true complexity, the learner will have access to a much
larger hypothesis space (capacity) than what is optimal. On the other
hand, if we underestimate the complexity, it is almost certain that we
will design an algorithm with insufficient capacity. This also extends
to the quantity and quality of training data that we are training with.
It is impossible, for example, to model an $n^\text{th}$ order process
with anything less than $n$ training samples without underfitting, and
this is without considering the possibility of measurement noise.

### Regularization

Generally speaking, regularisation is any modification we make to a
learning algorithm that is intended to reduce its generalisation error
but not its training error. Due to the large number of techniques that
fit this description within [ML]{acronym-label="ML"
acronym-form="singular+short"} and its subfields, it is outside the
scope of this paper to exhaustively cover this concept; however, the
core idea behind regularisation within [ML]{acronym-label="ML"
acronym-form="singular+short"} is motivated here through one of its
earliest forms.

A fundamental approach to regularization in [ML]{acronym-label="ML"
acronym-form="singular+short"} originating from *statistical learning*
is the use of **parameter norm penalties**. This technique aims at
guiding the parameter vectors through the addition of a penalty term in
the objective (cost/loss) function being minimised. This penalty
constrains (regularises) or shrinks the optimal
[w_vec]{acronym-label="w_vec" acronym-form="singular+short"} estimate
towards zero, effectively discouraging a learning algorithm from
producing an excessively complex model to avoid overfitting. Norm
penalty methods can be generalised as:

$$\gls{J_reg}(\glsvec{fn:param}; \gls{X}, \gls{Y})
    =
    \gls{J}(\glsvec{fn:param}; \gls{X}, \gls{Y})_\text{train}
    +
    \gls{ml:w_reg}\gls{o_reg}({\glsvec{fn:param}}),
    \label{eq:norm_penalty_reg}$$ $$\begin{aligned}
        \textrm{where, }
        \eqdesc{J_reg}\text{,} \\
        \eqdesc{ml:theta}\text{,} \\
        \eqdesc{X}\text{,} \\
        \eqdesc{Y}\text{,} \\
        \eqdesc{J}\text{,} \\
        \eqdesc{ml:w_reg}\text{,} \\
        \eqdesc{o_reg}\text{.} \\
    \end{aligned}$$

For learning algorithms which make use of gradient-based learning
techniques, the parameter gradient must be computed. This can be
challenging to compute in cases where the derivative of a cost function
is not easily determined, such as with [MAE]{acronym-label="MAE"
acronym-form="singular+short"} defined by . The expression of the
regularised cost function is given by,

$$\nabla_{\gls{w_vec}}\gls{J_reg}(\gls{w_vec};\gls{X},\gls{Y})
    =
    \nabla_{\gls{w_vec}}\gls{J}(\gls{w_vec};\gls{X},\gls{Y})
    +
    \gls{ml:w_reg}\nabla_{\gls{w_vec}}\gls{o_reg}(\glsvec{fn:param})\text{.}
    \label{eq:norm_penalty_reg_graident}$$

$\bm{\text{L}^2}$ **regularization**, also known as **ridge
regression**, **weight decay** or **Tikhonov regularization**
[@Goodfellow-et-al-2016 p. 227], is a parameter norm regularization
method which drives the weights of a model towards the origin in the
weight space and contributes to as,
$$\gls{o_reg}(\glsvec{fn:param})=\frac{1}{2}||\gls{w_vec}||^2_2.
    \label{eq:l2_reg}$$

Weight decay suppresses any irrelevant components of the weight vector
by choosing the smallest vector that solves the learning problem.
Furthermore, if the weight decay parameter is chosen correctly, noise
may be suppressed in the output, subsequently improving
generalization [@NIPS1991_8eefcfdf].

$\bm{\text{L}^1}$ **regularization**, also known as
**[LASSO]{acronym-label="LASSO" acronym-form="singular+short"}
regression** is a parameter norm regularization method which penalizes
the largest weight magnitudes in the weight vector. It is defined by,
$$\gls{o_reg}(\glsvec{fn:param})=||\gls{w_vec}||_1.
    \label{eq:l1_reg}$$

The key difference between $L^1$ and $L^2$ regularisation is that $L^1$
regularisation tends to better constrain values of $\gls{w_vec}$ to
zero, effectively performing feature selection on the inputs through
their weights. It can be less intuitive to interpret the $L^2$
regularisation as a feature selection method, as it can be unclear
whether the weights are truly being constrained to zero, or their
importance for generalisation is just less than other elements in
[w_vec]{acronym-label="w_vec" acronym-form="singular+short"}. This is
seen in , where the $L^1$ regularisation has reduced $w_1$ to zero, with
a negligible impact on the performance, whereas the $L^2$ variant has
not.

[\[fig:underfitting\]]{#fig:underfitting label="fig:underfitting"}

[\[fig:underfitting\]]{#fig:underfitting label="fig:underfitting"}

**Elastic net regularization** is a combination of both $L^1$ and $L^2$
regularization was introduced by Zou and Hastie [@ZouHastie2005] and is
expressed by:
$$\gls{o_reg}(\glsvec{fn:param})=\frac{1-\gls{b_ela}}{2}||\gls{w_vec}||^2_2 + \gls{b_ela}||\gls{w_vec}||_1\text{,}$$
$$\begin{aligned}
        \textrm{where, }
        \eqdesc{b_ela}\text{.} \\
    \end{aligned}$$ Looking again at , one might wonder if an
alternative vector norm regularization method would be more appropriate,
i.e. $p\notin\{1,2\}$ for . Hastie et al. state however, that experience
suggests that it is not worth the extra variance incurred to estimate
what vector norm $L^p$ should be used based on the
data [@hastie2009elements p. 73].

## Hyperparameters and Validation Sets

Most machine learning algorithms have parameters referred to as
hyperparameters. These define some algorithm settings which are not
self-adapted during learning. This, however, does not mean that the
algorithm cannot exist in a subroutine during which the hyperparameters
are optimised. An example of a hyperparameter is the value of
[ml:w_reg]{acronym-label="ml:w_reg" acronym-form="singular+short"}
chosen in . This parameter is kept constant throughout the learning
algorithm and heavily influences the resultant model. Optimising
hyperparameters on the training set that control capacity is generally
not appropriate. This is because the hyperparameter would tend towards
overfitting the training data, which would result in the best test
score, but with high variance and thus poor generalisation on unseen
data. Therefore as mentioned, the training procedure can be used as a
subroutine of a larger optimisation problem, which uses a validation set
to assess the performance of the model's hyperparameters.

[\[fig:ref-underfitting\]]{#fig:ref-underfitting
label="fig:ref-underfitting"}

[\[fig:reg-appropriate-capacity\]]{#fig:reg-appropriate-capacity
label="fig:reg-appropriate-capacity"}

[\[fig:overfitting\]]{#fig:overfitting label="fig:overfitting"}

The importance of optimising the hyperparameters of a learning
algorithm, especially relating to capacity, is illustrated in using the
example of regularization weight $\gls{ml:w_reg}$.

### The bias-variance tradeoff

Bias and variance measure two different sources in an estimator (model).
**The bias-variance tradeoff** is an inherent property of a model which
allows for the variance of the estimated parameter to be reduced by
increasing the bias of the model and vice versa. This property naturally
presents the **bias-variance dilemma** or **bias-variance problem**,
which is the conflict present in a supervised learning algorithm when it
attempts to generalise to new data outside of the training dataset. The
most common way to address this problem is through cross-validation.
Alternatively, the estimates of the model parameters $\hat{\theta}_m$
can be compared using [MSE]{acronym-label="MSE"
acronym-form="singular+short"}.

$$\begin{aligned}
        \text{\gls{MSE}}&=\gls{E}[(\hat{\theta}_m-\theta)^2]\\
        &=\text{Bias}(\hat{\theta}_m)^2+\text{Var}(\hat{\theta}_m)
    \end{aligned}$$

This concept of the bias-variance tradeoff is highly coupled to the
concepts of capacity, underfitting and overfitting, as evident in .

### Cross-Validation

Cross-validation is often used for the assessment and optimisation of
hyperparameters which improves a learning algorithm's ability to
generalise on unseen data. This is achieved through the separation of
training folds from a validation fold. The training fold is then
separated into training data and test data, while the validation fold is
left out of the training procedure entirely. The validation fold is then
used to assess the performance of the model on unseen data, which has
not been used to obtain the estimates of the model parameters. A common
way of performing cross-validation is to use a
[KFCV]{acronym-label="KFCV" acronym-form="singular+short"}, where $k$ is
the number of *approximately equal* folds that the data is separated
into. The total dataset is then iterated through, where each fold is
used once as the validation fold, and the remaining folds are used as
the training folds. The validation error is then calculated by averaging
the errors of all iterations. This is depicted in . [@hastie2009elements
p. 241-245]
