# search_term_2


- **Search Term:**
````python
"reinforcement learning" 
 AND "spacecraft" 
 AND ("guidance" OR "navigation" OR "control")
````

- **Total Hits:** 10
- **Papers per year:**
````python
2022 ▏    1 ████████▎
2021 ▏    3 █████████████████████████
2020 ▏    3 █████████████████████████
2019 ▏    1 ████████▎
2018 ▏    1 ████████▎
2017 ▏    1 ████████▎

````

- **To-do Review:**
- [ ] [1. Reinforcement Learning for Low-Thrust Trajectory Design of Interplanetary Missions (2020)](#1-Reinforcement-Learning-for-Low-Thrust-Trajectory-Design-of-Interplanetary-Missions-2020)
- [ ] [2. Adaptive Generalized ZEM-ZEV Feedback Guidance for Planetary Landing via a Deep Reinforcement Learning Approach (2020)](#2-Adaptive-Generalized-ZEM-ZEV-Feedback-Guidance-for-Planetary-Landing-via-a-Deep-Reinforcement-Learning-Approach-2020)
- [ ] [3. Autonomous Six-Degree-of-Freedom Spacecraft Docking Maneuvers via Reinforcement Learning (2020)](#3-Autonomous-Six-Degree-of-Freedom-Spacecraft-Docking-Maneuvers-via-Reinforcement-Learning-2020)
- [ ] [4. A Survey on Artificial Intelligence Trends in Spacecraft Guidance Dynamics and Control (2018)](#4-A-Survey-on-Artificial-Intelligence-Trends-in-Spacecraft-Guidance-Dynamics-and-Control-2018)
- [ ] [5. Hierarchical Reinforcement Learning Framework for Stochastic Spaceflight Campaign Design (2021)](#5-Hierarchical-Reinforcement-Learning-Framework-for-Stochastic-Spaceflight-Campaign-Design-2021)
- [ ] [6. Space Non-cooperative Object Active Tracking with Deep Reinforcement Learning (2021)](#6-Space-Non-cooperative-Object-Active-Tracking-with-Deep-Reinforcement-Learning-2021)
- [ ] [7. Scheduling the NASA Deep Space Network with Deep Reinforcement Learning (2021)](#7-Scheduling-the-NASA-Deep-Space-Network-with-Deep-Reinforcement-Learning-2021)
- [ ] [8. Space Navigator: a Tool for the Optimization of Collision Avoidance Maneuvers (2019)](#8-Space-Navigator:-a-Tool-for-the-Optimization-of-Collision-Avoidance-Maneuvers-2019)
- [ ] [9. FPGA Architecture for Deep Learning and its application to Planetary Robotics (2017)](#9-FPGA-Architecture-for-Deep-Learning-and-its-application-to-Planetary-Robotics-2017)
- [ ] [10. Learning Robust Policies for Generalized Debris Capture with an Automated Tether-Net System (2022)](#10-Learning-Robust-Policies-for-Generalized-Debris-Capture-with-an-Automated-Tether-Net-System-2022)


## 1. Reinforcement Learning for Low-Thrust Trajectory Design of Interplanetary Missions (2020)

<details>
<summary style="font-size:14px"><code>.bib</code></summary>
<p>

```bibtex
@article{2008.08501v1,
Author        = {Alessandro Zavoli and Lorenzo Federici},
Title         = {Reinforcement Learning for Low-Thrust Trajectory Design of
  Interplanetary Missions},
Eprint        = {2008.08501v1},
ArchivePrefix = {arXiv},
PrimaryClass  = {cs.LG},
Abstract      = {This paper investigates the use of Reinforcement Learning for the robust
design of low-thrust interplanetary trajectories in presence of severe
disturbances, modeled alternatively as Gaussian additive process noise,
observation noise, control actuation errors on thrust magnitude and direction,
and possibly multiple missed thrust events. The optimal control problem is
recast as a time-discrete Markov Decision Process to comply with the standard
formulation of reinforcement learning. An open-source implementation of the
state-of-the-art algorithm Proximal Policy Optimization is adopted to carry out
the training process of a deep neural network, used to map the spacecraft
(observed) states to the optimal control policy. The resulting Guidance and
Control Network provides both a robust nominal trajectory and the associated
closed-loop guidance law. Numerical results are presented for a typical
Earth-Mars mission. First, in order to validate the proposed approach, the
solution found in a (deterministic) unperturbed scenario is compared with the
optimal one provided by an indirect technique. Then, the robustness and
optimality of the obtained closed-loop guidance laws is assessed by means of
Monte Carlo campaigns performed in the considered uncertain scenarios. These
preliminary results open up new horizons for the use of reinforcement learning
in the robust design of interplanetary missions.},
Year          = {2020},
Month         = {Aug},
Url           = {http://arxiv.org/abs/2008.08501v1},
File          = {2008.08501v1.pdf}
}
```

</p></details>

[Download PDF](http://arxiv.org/pdf/2008.08501v1)

This paper investigates the use of Reinforcement Learning for the robust design of low-thrust interplanetary trajectories in presence of severe disturbances, modeled alternatively as Gaussian additive process noise, observation noise, control actuation errors on thrust magnitude and direction, and possibly multiple missed thrust events. The optimal control problem is recast as a time-discrete Markov Decision Process to comply with the standard formulation of reinforcement learning. An open-source implementation of the state-of-the-art algorithm Proximal Policy Optimization is adopted to carry out the training process of a deep neural network, used to map the spacecraft (observed) states to the optimal control policy. The resulting Guidance and Control Network provides both a robust nominal trajectory and the associated closed-loop guidance law. Numerical results are presented for a typical Earth-Mars mission. First, in order to validate the proposed approach, the solution found in a (deterministic) unperturbed scenario is compared with the optimal one provided by an indirect technique. Then, the robustness and optimality of the obtained closed-loop guidance laws is assessed by means of Monte Carlo campaigns performed in the considered uncertain scenarios. These preliminary results open up new horizons for the use of reinforcement learning in the robust design of interplanetary missions.

- [ ] Relevant
- [ ] Irrelevant

## 2. Adaptive Generalized ZEM-ZEV Feedback Guidance for Planetary Landing via a Deep Reinforcement Learning Approach (2020)

<details>
<summary style="font-size:14px"><code>.bib</code></summary>
<p>

```bibtex
@article{2003.02182v1,
Author        = {Roberto Furfaro and Andrea Scorsoglio and Richard Linares and Mauro Massari},
Title         = {Adaptive Generalized ZEM-ZEV Feedback Guidance for Planetary Landing via
  a Deep Reinforcement Learning Approach},
Eprint        = {2003.02182v1},
DOI           = {10.1016/j.actaastro.2020.02.051},
ArchivePrefix = {arXiv},
PrimaryClass  = {eess.SY},
Abstract      = {Precision landing on large and small planetary bodies is a technology of
utmost importance for future human and robotic exploration of the solar system.
In this context, the Zero-Effort-Miss/Zero-Effort-Velocity (ZEM/ZEV) feedback
guidance algorithm has been studied extensively and is still a field of active
research. The algorithm, although powerful in terms of accuracy and ease of
implementation, has some limitations. Therefore with this paper we present an
adaptive guidance algorithm based on classical ZEM/ZEV in which machine
learning is used to overcome its limitations and create a closed loop guidance
algorithm that is sufficiently lightweight to be implemented on board
spacecraft and flexible enough to be able to adapt to the given constraint
scenario. The adopted methodology is an actor-critic reinforcement learning
algorithm that learns the parameters of the above-mentioned guidance
architecture according to the given problem constraints.},
Year          = {2020},
Month         = {Mar},
Url           = {http://arxiv.org/abs/2003.02182v1},
File          = {2003.02182v1.pdf}
}
```

</p></details>

[Download PDF](http://arxiv.org/pdf/2003.02182v1)

Precision landing on large and small planetary bodies is a technology of utmost importance for future human and robotic exploration of the solar system. In this context, the Zero-Effort-Miss/Zero-Effort-Velocity (ZEM/ZEV) feedback guidance algorithm has been studied extensively and is still a field of active research. The algorithm, although powerful in terms of accuracy and ease of implementation, has some limitations. Therefore with this paper we present an adaptive guidance algorithm based on classical ZEM/ZEV in which machine learning is used to overcome its limitations and create a closed loop guidance algorithm that is sufficiently lightweight to be implemented on board spacecraft and flexible enough to be able to adapt to the given constraint scenario. The adopted methodology is an actor-critic reinforcement learning algorithm that learns the parameters of the above-mentioned guidance architecture according to the given problem constraints.

- [ ] Relevant
- [ ] Irrelevant

## 3. Autonomous Six-Degree-of-Freedom Spacecraft Docking Maneuvers via Reinforcement Learning (2020)

<details>
<summary style="font-size:14px"><code>.bib</code></summary>
<p>

```bibtex
@article{2008.03215v1,
Author        = {Charles E. Oestreich and Richard Linares and Ravi Gondhalekar},
Title         = {Autonomous Six-Degree-of-Freedom Spacecraft Docking Maneuvers via
  Reinforcement Learning},
Eprint        = {2008.03215v1},
ArchivePrefix = {arXiv},
PrimaryClass  = {eess.SY},
Abstract      = {A policy for six-degree-of-freedom docking maneuvers is developed through
reinforcement learning and implemented as a feedback control law. Reinforcement
learning provides a potential framework for robust, autonomous maneuvers in
uncertain environments with low on-board computational cost. Specifically,
proximal policy optimization is used to produce a docking policy that is valid
over a portion of the six-degree-of-freedom state-space while striving to
minimize performance and control costs. Experiments using the simulated Apollo
transposition and docking maneuver exhibit the policy's capabilities and
provide a comparison with standard optimal control techniques. Furthermore,
specific challenges and work-arounds, as well as a discussion on the benefits
and disadvantages of reinforcement learning for docking policies, are discussed
to facilitate future research. As such, this work will serve as a foundation
for further investigation of learning-based control laws for spacecraft
proximity operations in uncertain environments.},
Year          = {2020},
Month         = {Aug},
Url           = {http://arxiv.org/abs/2008.03215v1},
File          = {2008.03215v1.pdf}
}
```

</p></details>

[Download PDF](http://arxiv.org/pdf/2008.03215v1)

A policy for six-degree-of-freedom docking maneuvers is developed through reinforcement learning and implemented as a feedback control law. Reinforcement learning provides a potential framework for robust, autonomous maneuvers in uncertain environments with low on-board computational cost. Specifically, proximal policy optimization is used to produce a docking policy that is valid over a portion of the six-degree-of-freedom state-space while striving to minimize performance and control costs. Experiments using the simulated Apollo transposition and docking maneuver exhibit the policy's capabilities and provide a comparison with standard optimal control techniques. Furthermore, specific challenges and work-arounds, as well as a discussion on the benefits and disadvantages of reinforcement learning for docking policies, are discussed to facilitate future research. As such, this work will serve as a foundation for further investigation of learning-based control laws for spacecraft proximity operations in uncertain environments.

- [ ] Relevant
- [ ] Irrelevant

## 4. A Survey on Artificial Intelligence Trends in Spacecraft Guidance Dynamics and Control (2018)

<details>
<summary style="font-size:14px"><code>.bib</code></summary>
<p>

```bibtex
@article{1812.02948v1,
Author        = {Dario Izzo and Marcus Märtens and Binfeng Pan},
Title         = {A Survey on Artificial Intelligence Trends in Spacecraft Guidance
  Dynamics and Control},
Eprint        = {1812.02948v1},
ArchivePrefix = {arXiv},
PrimaryClass  = {cs.NE},
Abstract      = {The rapid developments of Artificial Intelligence in the last decade are
influencing Aerospace Engineering to a great extent and research in this
context is proliferating. We share our observations on the recent developments
in the area of Spacecraft Guidance Dynamics and Control, giving selected
examples on success stories that have been motivated by mission designs. Our
focus is on evolutionary optimisation, tree searches and machine learning,
including deep learning and reinforcement learning as the key technologies and
drivers for current and future research in the field. From a high-level
perspective, we survey various scenarios for which these approaches have been
successfully applied or are under strong scientific investigation. Whenever
possible, we highlight the relations and synergies that can be obtained by
combining different techniques and projects towards future domains for which
newly emerging artificial intelligence techniques are expected to become game
changers.},
Year          = {2018},
Month         = {Dec},
Url           = {http://arxiv.org/abs/1812.02948v1},
File          = {1812.02948v1.pdf}
}
```

</p></details>

[Download PDF](http://arxiv.org/pdf/1812.02948v1)

The rapid developments of Artificial Intelligence in the last decade are influencing Aerospace Engineering to a great extent and research in this context is proliferating. We share our observations on the recent developments in the area of Spacecraft Guidance Dynamics and Control, giving selected examples on success stories that have been motivated by mission designs. Our focus is on evolutionary optimisation, tree searches and machine learning, including deep learning and reinforcement learning as the key technologies and drivers for current and future research in the field. From a high-level perspective, we survey various scenarios for which these approaches have been successfully applied or are under strong scientific investigation. Whenever possible, we highlight the relations and synergies that can be obtained by combining different techniques and projects towards future domains for which newly emerging artificial intelligence techniques are expected to become game changers.

- [ ] Relevant
- [ ] Irrelevant

## 5. Hierarchical Reinforcement Learning Framework for Stochastic Spaceflight Campaign Design (2021)

<details>
<summary style="font-size:14px"><code>.bib</code></summary>
<p>

```bibtex
@article{2103.08981v2,
Author        = {Yuji Takubo and Hao Chen and Koki Ho},
Title         = {Hierarchical Reinforcement Learning Framework for Stochastic Spaceflight
  Campaign Design},
Eprint        = {2103.08981v2},
DOI           = {10.2514/1.A35122},
ArchivePrefix = {arXiv},
PrimaryClass  = {cs.LG},
Abstract      = {This paper develops a hierarchical reinforcement learning architecture for
multimission spaceflight campaign design under uncertainty, including vehicle
design, infrastructure deployment planning, and space transportation
scheduling. This problem involves a high-dimensional design space and is
challenging especially with uncertainty present. To tackle this challenge, the
developed framework has a hierarchical structure with reinforcement learning
and network-based mixed-integer linear programming (MILP), where the former
optimizes campaign-level decisions (e.g., design of the vehicle used throughout
the campaign, destination demand assigned to each mission in the campaign),
whereas the latter optimizes the detailed mission-level decisions (e.g., when
to launch what from where to where). The framework is applied to a set of human
lunar exploration campaign scenarios with uncertain in situ resource
utilization performance as a case study. The main value of this work is its
integration of the rapidly growing reinforcement learning research and the
existing MILP-based space logistics methods through a hierarchical framework to
handle the otherwise intractable complexity of space mission design under
uncertainty. This unique framework is expected to be a critical steppingstone
for the emerging research direction of artificial intelligence for space
mission design.},
Year          = {2021},
Month         = {Mar},
Note          = {Journal of Spacecraft and Rockets (Articles in Advance), 2022},
Url           = {http://arxiv.org/abs/2103.08981v2},
File          = {2103.08981v2.pdf}
}
```

</p></details>

[Download PDF](http://arxiv.org/pdf/2103.08981v2)

This paper develops a hierarchical reinforcement learning architecture for multimission spaceflight campaign design under uncertainty, including vehicle design, infrastructure deployment planning, and space transportation scheduling. This problem involves a high-dimensional design space and is challenging especially with uncertainty present. To tackle this challenge, the developed framework has a hierarchical structure with reinforcement learning and network-based mixed-integer linear programming (MILP), where the former optimizes campaign-level decisions (e.g., design of the vehicle used throughout the campaign, destination demand assigned to each mission in the campaign), whereas the latter optimizes the detailed mission-level decisions (e.g., when to launch what from where to where). The framework is applied to a set of human lunar exploration campaign scenarios with uncertain in situ resource utilization performance as a case study. The main value of this work is its integration of the rapidly growing reinforcement learning research and the existing MILP-based space logistics methods through a hierarchical framework to handle the otherwise intractable complexity of space mission design under uncertainty. This unique framework is expected to be a critical steppingstone for the emerging research direction of artificial intelligence for space mission design.

- [ ] Relevant
- [ ] Irrelevant

## 6. Space Non-cooperative Object Active Tracking with Deep Reinforcement Learning (2021)

<details>
<summary style="font-size:14px"><code>.bib</code></summary>
<p>

```bibtex
@article{2112.09854v1,
Author        = {Dong Zhou and Guanghui Sun and Wenxiao Lei},
Title         = {Space Non-cooperative Object Active Tracking with Deep Reinforcement
  Learning},
Eprint        = {2112.09854v1},
ArchivePrefix = {arXiv},
PrimaryClass  = {cs.CV},
Abstract      = {Active visual tracking of space non-cooperative object is significant for
future intelligent spacecraft to realise space debris removal, asteroid
exploration, autonomous rendezvous and docking. However, existing works often
consider this task into different subproblems (e.g. image preprocessing,
feature extraction and matching, position and pose estimation, control law
design) and optimize each module alone, which are trivial and sub-optimal. To
this end, we propose an end-to-end active visual tracking method based on DQN
algorithm, named as DRLAVT. It can guide the chasing spacecraft approach to
arbitrary space non-cooperative target merely relied on color or RGBD images,
which significantly outperforms position-based visual servoing baseline
algorithm that adopts state-of-the-art 2D monocular tracker, SiamRPN. Extensive
experiments implemented with diverse network architectures, different
perturbations and multiple targets demonstrate the advancement and robustness
of DRLAVT. In addition, We further prove our method indeed learnt the motion
patterns of target with deep reinforcement learning through hundreds of
trial-and-errors.},
Year          = {2021},
Month         = {Dec},
Url           = {http://arxiv.org/abs/2112.09854v1},
File          = {2112.09854v1.pdf}
}
```

</p></details>

[Download PDF](http://arxiv.org/pdf/2112.09854v1)

Active visual tracking of space non-cooperative object is significant for future intelligent spacecraft to realise space debris removal, asteroid exploration, autonomous rendezvous and docking. However, existing works often consider this task into different subproblems (e.g. image preprocessing, feature extraction and matching, position and pose estimation, control law design) and optimize each module alone, which are trivial and sub-optimal. To this end, we propose an end-to-end active visual tracking method based on DQN algorithm, named as DRLAVT. It can guide the chasing spacecraft approach to arbitrary space non-cooperative target merely relied on color or RGBD images, which significantly outperforms position-based visual servoing baseline algorithm that adopts state-of-the-art 2D monocular tracker, SiamRPN. Extensive experiments implemented with diverse network architectures, different perturbations and multiple targets demonstrate the advancement and robustness of DRLAVT. In addition, We further prove our method indeed learnt the motion patterns of target with deep reinforcement learning through hundreds of trial-and-errors.

- [ ] Relevant
- [ ] Irrelevant

## 7. Scheduling the NASA Deep Space Network with Deep Reinforcement Learning (2021)

<details>
<summary style="font-size:14px"><code>.bib</code></summary>
<p>

```bibtex
@article{2102.05167v1,
Author        = {Edwin Goh and Hamsa Shwetha Venkataram and Mark Hoffmann and Mark Johnston and Brian Wilson},
Title         = {Scheduling the NASA Deep Space Network with Deep Reinforcement Learning},
Eprint        = {2102.05167v1},
ArchivePrefix = {arXiv},
PrimaryClass  = {cs.LG},
Abstract      = {With three complexes spread evenly across the Earth, NASA's Deep Space
Network (DSN) is the primary means of communications as well as a significant
scientific instrument for dozens of active missions around the world. A rapidly
rising number of spacecraft and increasingly complex scientific instruments
with higher bandwidth requirements have resulted in demand that exceeds the
network's capacity across its 12 antennae. The existing DSN scheduling process
operates on a rolling weekly basis and is time-consuming; for a given week,
generation of the final baseline schedule of spacecraft tracking passes takes
roughly 5 months from the initial requirements submission deadline, with
several weeks of peer-to-peer negotiations in between. This paper proposes a
deep reinforcement learning (RL) approach to generate candidate DSN schedules
from mission requests and spacecraft ephemeris data with demonstrated
capability to address real-world operational constraints. A deep RL agent is
developed that takes mission requests for a given week as input, and interacts
with a DSN scheduling environment to allocate tracks such that its reward
signal is maximized. A comparison is made between an agent trained using
Proximal Policy Optimization and its random, untrained counterpart. The results
represent a proof-of-concept that, given a well-shaped reward signal, a deep RL
agent can learn the complex heuristics used by experts to schedule the DSN. A
trained agent can potentially be used to generate candidate schedules to
bootstrap the scheduling process and thus reduce the turnaround cycle for DSN
scheduling.},
Year          = {2021},
Month         = {Feb},
Url           = {http://arxiv.org/abs/2102.05167v1},
File          = {2102.05167v1.pdf}
}
```

</p></details>

[Download PDF](http://arxiv.org/pdf/2102.05167v1)

With three complexes spread evenly across the Earth, NASA's Deep Space Network (DSN) is the primary means of communications as well as a significant scientific instrument for dozens of active missions around the world. A rapidly rising number of spacecraft and increasingly complex scientific instruments with higher bandwidth requirements have resulted in demand that exceeds the network's capacity across its 12 antennae. The existing DSN scheduling process operates on a rolling weekly basis and is time-consuming; for a given week, generation of the final baseline schedule of spacecraft tracking passes takes roughly 5 months from the initial requirements submission deadline, with several weeks of peer-to-peer negotiations in between. This paper proposes a deep reinforcement learning (RL) approach to generate candidate DSN schedules from mission requests and spacecraft ephemeris data with demonstrated capability to address real-world operational constraints. A deep RL agent is developed that takes mission requests for a given week as input, and interacts with a DSN scheduling environment to allocate tracks such that its reward signal is maximized. A comparison is made between an agent trained using Proximal Policy Optimization and its random, untrained counterpart. The results represent a proof-of-concept that, given a well-shaped reward signal, a deep RL agent can learn the complex heuristics used by experts to schedule the DSN. A trained agent can potentially be used to generate candidate schedules to bootstrap the scheduling process and thus reduce the turnaround cycle for DSN scheduling.

- [ ] Relevant
- [ ] Irrelevant

## 8. Space Navigator: a Tool for the Optimization of Collision Avoidance Maneuvers (2019)

<details>
<summary style="font-size:14px"><code>.bib</code></summary>
<p>

```bibtex
@article{1902.02095v1,
Author        = {Leonid Gremyachikh and Dmitrii Dubov and Nikita Kazeev and Andrey Kulibaba and Andrey Skuratov and Anton Tereshkin and Andrey Ustyuzhanin and Lubov Shiryaeva and Sergej Shishkin},
Title         = {Space Navigator: a Tool for the Optimization of Collision Avoidance
  Maneuvers},
Eprint        = {1902.02095v1},
ArchivePrefix = {arXiv},
PrimaryClass  = {cs.SY},
Abstract      = {The number of space objects will grow several times in a few years due to the
planned launches of constellations of thousands microsatellites. It leads to a
significant increase in the threat of satellite collisions. Spacecraft must
undertake collision avoidance maneuvers to mitigate the risk. According to
publicly available information, conjunction events are now manually handled by
operators on the Earth. The manual maneuver planning requires qualified
personnel and will be impractical for constellations of thousands satellites.
In this paper we propose a new modular autonomous collision avoidance system
called "Space Navigator". It is based on a novel maneuver optimization approach
that combines domain knowledge with Reinforcement Learning methods.},
Year          = {2019},
Month         = {Feb},
Note          = {Advances in the Astronautical Sciences 2020 First IAA/AAS SciTech
  Forum on Space Flight Mechanics and Space Structures and Materials
  Conference, volume 170},
Url           = {http://arxiv.org/abs/1902.02095v1},
File          = {1902.02095v1.pdf}
}
```

</p></details>

[Download PDF](http://arxiv.org/pdf/1902.02095v1)

The number of space objects will grow several times in a few years due to the planned launches of constellations of thousands microsatellites. It leads to a significant increase in the threat of satellite collisions. Spacecraft must undertake collision avoidance maneuvers to mitigate the risk. According to publicly available information, conjunction events are now manually handled by operators on the Earth. The manual maneuver planning requires qualified personnel and will be impractical for constellations of thousands satellites. In this paper we propose a new modular autonomous collision avoidance system called "Space Navigator". It is based on a novel maneuver optimization approach that combines domain knowledge with Reinforcement Learning methods.

- [ ] Relevant
- [ ] Irrelevant

## 9. FPGA Architecture for Deep Learning and its application to Planetary Robotics (2017)

<details>
<summary style="font-size:14px"><code>.bib</code></summary>
<p>

```bibtex
@article{1701.07543v1,
Author        = {Pranay Gankidi and Jekan Thangavelautham},
Title         = {FPGA Architecture for Deep Learning and its application to Planetary
  Robotics},
Eprint        = {1701.07543v1},
ArchivePrefix = {arXiv},
PrimaryClass  = {cs.LG},
Abstract      = {Autonomous control systems onboard planetary rovers and spacecraft benefit
from having cognitive capabilities like learning so that they can adapt to
unexpected situations in-situ. Q-learning is a form of reinforcement learning
and it has been efficient in solving certain class of learning problems.
However, embedded systems onboard planetary rovers and spacecraft rarely
implement learning algorithms due to the constraints faced in the field, like
processing power, chip size, convergence rate and costs due to the need for
radiation hardening. These challenges present a compelling need for a portable,
low-power, area efficient hardware accelerator to make learning algorithms
practical onboard space hardware. This paper presents a FPGA implementation of
Q-learning with Artificial Neural Networks (ANN). This method matches the
massive parallelism inherent in neural network software with the fine-grain
parallelism of an FPGA hardware thereby dramatically reducing processing time.
Mars Science Laboratory currently uses Xilinx-Space-grade Virtex FPGA devices
for image processing, pyrotechnic operation control and obstacle avoidance. We
simulate and program our architecture on a Xilinx Virtex 7 FPGA. The
architectural implementation for a single neuron Q-learning and a more complex
Multilayer Perception (MLP) Q-learning accelerator has been demonstrated. The
results show up to a 43-fold speed up by Virtex 7 FPGAs compared to a
conventional Intel i5 2.3 GHz CPU. Finally, we simulate the proposed
architecture using the Symphony simulator and compiler from Xilinx, and
evaluate the performance and power consumption.},
Year          = {2017},
Month         = {Jan},
Url           = {http://arxiv.org/abs/1701.07543v1},
File          = {1701.07543v1.pdf}
}
```

</p></details>

[Download PDF](http://arxiv.org/pdf/1701.07543v1)

Autonomous control systems onboard planetary rovers and spacecraft benefit from having cognitive capabilities like learning so that they can adapt to unexpected situations in-situ. Q-learning is a form of reinforcement learning and it has been efficient in solving certain class of learning problems. However, embedded systems onboard planetary rovers and spacecraft rarely implement learning algorithms due to the constraints faced in the field, like processing power, chip size, convergence rate and costs due to the need for radiation hardening. These challenges present a compelling need for a portable, low-power, area efficient hardware accelerator to make learning algorithms practical onboard space hardware. This paper presents a FPGA implementation of Q-learning with Artificial Neural Networks (ANN). This method matches the massive parallelism inherent in neural network software with the fine-grain parallelism of an FPGA hardware thereby dramatically reducing processing time. Mars Science Laboratory currently uses Xilinx-Space-grade Virtex FPGA devices for image processing, pyrotechnic operation control and obstacle avoidance. We simulate and program our architecture on a Xilinx Virtex 7 FPGA. The architectural implementation for a single neuron Q-learning and a more complex Multilayer Perception (MLP) Q-learning accelerator has been demonstrated. The results show up to a 43-fold speed up by Virtex 7 FPGAs compared to a conventional Intel i5 2.3 GHz CPU. Finally, we simulate the proposed architecture using the Symphony simulator and compiler from Xilinx, and evaluate the performance and power consumption.

- [ ] Relevant
- [ ] Irrelevant

## 10. Learning Robust Policies for Generalized Debris Capture with an Automated Tether-Net System (2022)

<details>
<summary style="font-size:14px"><code>.bib</code></summary>
<p>

```bibtex
@article{2201.04180v1,
Author        = {Chen Zeng and Grant Hecht and Prajit KrisshnaKumar and Raj K. Shah and Souma Chowdhury and Eleonora M. Botta},
Title         = {Learning Robust Policies for Generalized Debris Capture with an
  Automated Tether-Net System},
Eprint        = {2201.04180v1},
DOI           = {10.2514/6.2022-2379},
ArchivePrefix = {arXiv},
PrimaryClass  = {cs.RO},
Abstract      = {Tether-net launched from a chaser spacecraft provides a promising method to
capture and dispose of large space debris in orbit. This tether-net system is
subject to several sources of uncertainty in sensing and actuation that affect
the performance of its net launch and closing control. Earlier
reliability-based optimization approaches to design control actions however
remain challenging and computationally prohibitive to generalize over varying
launch scenarios and target (debris) state relative to the chaser. To search
for a general and reliable control policy, this paper presents a reinforcement
learning framework that integrates a proximal policy optimization (PPO2)
approach with net dynamics simulations. The latter allows evaluating the
episodes of net-based target capture, and estimate the capture quality index
that serves as the reward feedback to PPO2. Here, the learned policy is
designed to model the timing of the net closing action based on the state of
the moving net and the target, under any given launch scenario. A stochastic
state transition model is considered in order to incorporate synthetic
uncertainties in state estimation and launch actuation. Along with notable
reward improvement during training, the trained policy demonstrates capture
performance (over a wide range of launch/target scenarios) that is close to
that obtained with reliability-based optimization run over an individual
scenario.},
Year          = {2022},
Month         = {Jan},
Url           = {http://arxiv.org/abs/2201.04180v1},
File          = {2201.04180v1.pdf}
}
```

</p></details>

[Download PDF](http://arxiv.org/pdf/2201.04180v1)

Tether-net launched from a chaser spacecraft provides a promising method to capture and dispose of large space debris in orbit. This tether-net system is subject to several sources of uncertainty in sensing and actuation that affect the performance of its net launch and closing control. Earlier reliability-based optimization approaches to design control actions however remain challenging and computationally prohibitive to generalize over varying launch scenarios and target (debris) state relative to the chaser. To search for a general and reliable control policy, this paper presents a reinforcement learning framework that integrates a proximal policy optimization (PPO2) approach with net dynamics simulations. The latter allows evaluating the episodes of net-based target capture, and estimate the capture quality index that serves as the reward feedback to PPO2. Here, the learned policy is designed to model the timing of the net closing action based on the state of the moving net and the target, under any given launch scenario. A stochastic state transition model is considered in order to incorporate synthetic uncertainties in state estimation and launch actuation. Along with notable reward improvement during training, the trained policy demonstrates capture performance (over a wide range of launch/target scenarios) that is close to that obtained with reliability-based optimization run over an individual scenario.

- [ ] Relevant
- [ ] Irrelevant

