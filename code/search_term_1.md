# search_term_1


- **Search Term:**
````python
"machine learning" 
 AND "spacecraft" 
 AND ("asteroid" OR "comet" OR "small body" OR "minor planet") 
 AND ("characterisation" OR "characterization")
````

- **Total Hits:** 5
- **Papers per year:**
````python
2021 ▏    1 █████████████████████████
2020 ▏    0 ▏
2019 ▏    1 █████████████████████████
2018 ▏    0 ▏
2017 ▏    1 █████████████████████████
2016 ▏    1 █████████████████████████
2015 ▏    0 ▏
2014 ▏    1 █████████████████████████

````

- **To-do Review:**
- [ ] [1. Network of Nano-Landers for In-Situ Characterization of Asteroid Impact Studies (2017)](#1-Network-of-Nano-Landers-for-In-Situ-Characterization-of-Asteroid-Impact-Studies-2017)
- [ ] [2. Seeker based Adaptive Guidance via Reinforcement Meta-Learning Applied to Asteroid Close Proximity Operations (2019)](#2-Seeker-based-Adaptive-Guidance-via-Reinforcement-Meta-Learning-Applied-to-Asteroid-Close-Proximity-Operations-2019)
- [ ] [3. Automated Real-Time Classification and Decision Making in Massive Data Streams from Synoptic Sky Surveys (2014)](#3-Automated-Real-Time-Classification-and-Decision-Making-in-Massive-Data-Streams-from-Synoptic-Sky-Surveys-2014)
- [ ] [4. Real-Time Data Mining of Massive Data Streams from Synoptic Sky Surveys (2016)](#4-Real-Time-Data-Mining-of-Massive-Data-Streams-from-Synoptic-Sky-Surveys-2016)
- [ ] [5. Bennu's global surface and two candidate sample sites characterized by spectral clustering of OSIRIS-REx multispectral images (2021)](#5-Bennus-global-surface-and-two-candidate-sample-sites-characterized-by-spectral-clustering-of-OSIRIS-REx-multispectral-images-2021)


## 1. Network of Nano-Landers for In-Situ Characterization of Asteroid Impact Studies (2017)

<details>
<summary style="font-size:14px"><code>.bib</code></summary>
<p>

```bibtex
@article{1709.02885v1,
Author        = {Himangshu Kalita and Erik Asphaug and Stephen Schwartz and Jekanthan Thangavelautham},
Title         = {Network of Nano-Landers for In-Situ Characterization of Asteroid Impact
  Studies},
Eprint        = {1709.02885v1},
ArchivePrefix = {arXiv},
PrimaryClass  = {cs.RO},
Abstract      = {Exploration of asteroids and comets can give insight into the origins of the
solar system and can be instrumental in planetary defence and in-situ resource
utilization (ISRU). Asteroids, due to their low gravity are a challenging
target for surface exploration. Current missions envision performing
touch-and-go operations over an asteroid surface. In this work, we analyse the
feasibility of sending scores of nano-landers, each 1 kg in mass and volume of
1U, or 1000 cm3. These landers would hop, roll and fly over the asteroid
surface. The landers would include science instruments such as stereo cameras,
hand-lens imagers and spectrometers to characterize rock composition. A network
of nano-landers situated on the surface of an asteroid can provide unique and
very detailed measurements of a spacecraft impacting onto an asteroid surface.
A full-scale, artificial impact experiment onto an asteroid can help
characterize its composition and geology and help in the development of
asteroid deflection techniques intended for planetary defence. Scores of
nano-landers could provide multiple complementary views of the impact,
resultant seismic activity and trajectory of the ejecta. The nano-landers can
analyse the pristine, unearthed regolith shielded from effects of UV and cosmic
rays and that may be millions of years old. Our approach to formulating this
mission concepts utilizes automated machine learning techniques in the planning
and design of space systems. We use a form of Darwinian selection to select and
identify suitable number of nano-landers, the on-board instruments and control
system to explore and navigate the asteroid environment. Scenarios are
generated in simulation and evaluated against quantifiable mission goals such
as area explored on the asteroid and amount of data recorded from the impact
event.},
Year          = {2017},
Month         = {Sep},
Url           = {http://arxiv.org/abs/1709.02885v1},
File          = {1709.02885v1.pdf}
}
```

</p></details>

[Download PDF](http://arxiv.org/pdf/1709.02885v1)

Exploration of asteroids and comets can give insight into the origins of the solar system and can be instrumental in planetary defence and in-situ resource utilization (ISRU). Asteroids, due to their low gravity are a challenging target for surface exploration. Current missions envision performing touch-and-go operations over an asteroid surface. In this work, we analyse the feasibility of sending scores of nano-landers, each 1 kg in mass and volume of 1U, or 1000 cm3. These landers would hop, roll and fly over the asteroid surface. The landers would include science instruments such as stereo cameras, hand-lens imagers and spectrometers to characterize rock composition. A network of nano-landers situated on the surface of an asteroid can provide unique and very detailed measurements of a spacecraft impacting onto an asteroid surface. A full-scale, artificial impact experiment onto an asteroid can help characterize its composition and geology and help in the development of asteroid deflection techniques intended for planetary defence. Scores of nano-landers could provide multiple complementary views of the impact, resultant seismic activity and trajectory of the ejecta. The nano-landers can analyse the pristine, unearthed regolith shielded from effects of UV and cosmic rays and that may be millions of years old. Our approach to formulating this mission concepts utilizes automated machine learning techniques in the planning and design of space systems. We use a form of Darwinian selection to select and identify suitable number of nano-landers, the on-board instruments and control system to explore and navigate the asteroid environment. Scenarios are generated in simulation and evaluated against quantifiable mission goals such as area explored on the asteroid and amount of data recorded from the impact event.

- [ ] Relevant
- [ ] Irrelevant

## 2. Seeker based Adaptive Guidance via Reinforcement Meta-Learning Applied to Asteroid Close Proximity Operations (2019)

<details>
<summary style="font-size:14px"><code>.bib</code></summary>
<p>

```bibtex
@article{1907.06098v1,
Author        = {Brian Gaudet and Richard Linares and Roberto Furfaro},
Title         = {Seeker based Adaptive Guidance via Reinforcement Meta-Learning Applied
  to Asteroid Close Proximity Operations},
Eprint        = {1907.06098v1},
DOI           = {10.1016/j.actaastro.2020.02.036},
ArchivePrefix = {arXiv},
PrimaryClass  = {eess.SY},
Abstract      = {Current practice for asteroid close proximity maneuvers requires extremely
accurate characterization of the environmental dynamics and precise spacecraft
positioning prior to the maneuver. This creates a delay of several months
between the spacecraft's arrival and the ability to safely complete close
proximity maneuvers. In this work we develop an adaptive integrated guidance,
navigation, and control system that can complete these maneuvers in
environments with unknown dynamics, with initial conditions spanning a large
deployment region, and without a shape model of the asteroid. The system is
implemented as a policy optimized using reinforcement meta-learning. The
spacecraft is equipped with an optical seeker that locks to either a terrain
feature, back-scattered light from a targeting laser, or an active beacon, and
the policy maps observations consisting of seeker angles and LIDAR range
readings directly to engine thrust commands. The policy implements a recurrent
network layer that allows the deployed policy to adapt real time to both
environmental forces acting on the agent and internal disturbances such as
actuator failure and center of mass variation. We validate the guidance system
through simulated landing maneuvers in a six degrees-of-freedom simulator. The
simulator randomizes the asteroid's characteristics such as solar radiation
pressure, density, spin rate, and nutation angle, requiring the guidance and
control system to adapt to the environment. We also demonstrate robustness to
actuator failure, sensor bias, and changes in the spacecraft's center of mass
and inertia tensor. Finally, we suggest a concept of operations for asteroid
close proximity maneuvers that is compatible with the guidance system.},
Year          = {2019},
Month         = {Jul},
Url           = {http://arxiv.org/abs/1907.06098v1},
File          = {1907.06098v1.pdf}
}
```

</p></details>

[Download PDF](http://arxiv.org/pdf/1907.06098v1)

Current practice for asteroid close proximity maneuvers requires extremely accurate characterization of the environmental dynamics and precise spacecraft positioning prior to the maneuver. This creates a delay of several months between the spacecraft's arrival and the ability to safely complete close proximity maneuvers. In this work we develop an adaptive integrated guidance, navigation, and control system that can complete these maneuvers in environments with unknown dynamics, with initial conditions spanning a large deployment region, and without a shape model of the asteroid. The system is implemented as a policy optimized using reinforcement meta-learning. The spacecraft is equipped with an optical seeker that locks to either a terrain feature, back-scattered light from a targeting laser, or an active beacon, and the policy maps observations consisting of seeker angles and LIDAR range readings directly to engine thrust commands. The policy implements a recurrent network layer that allows the deployed policy to adapt real time to both environmental forces acting on the agent and internal disturbances such as actuator failure and center of mass variation. We validate the guidance system through simulated landing maneuvers in a six degrees-of-freedom simulator. The simulator randomizes the asteroid's characteristics such as solar radiation pressure, density, spin rate, and nutation angle, requiring the guidance and control system to adapt to the environment. We also demonstrate robustness to actuator failure, sensor bias, and changes in the spacecraft's center of mass and inertia tensor. Finally, we suggest a concept of operations for asteroid close proximity maneuvers that is compatible with the guidance system.

- [ ] Relevant
- [ ] Irrelevant

## 3. Automated Real-Time Classification and Decision Making in Massive Data Streams from Synoptic Sky Surveys (2014)

<details>
<summary style="font-size:14px"><code>.bib</code></summary>
<p>

```bibtex
@article{1407.3502v1,
Author        = {S. G. Djorgovski and A. A. Mahabal and C. Donalek and M. J. Graham and A. J. Drake and M. Turmon and T. Fuchs},
Title         = {Automated Real-Time Classification and Decision Making in Massive Data
  Streams from Synoptic Sky Surveys},
Eprint        = {1407.3502v1},
DOI           = {10.1109/eScience.2014.7},
ArchivePrefix = {arXiv},
PrimaryClass  = {astro-ph.IM},
Abstract      = {The nature of scientific and technological data collection is evolving
rapidly: data volumes and rates grow exponentially, with increasing complexity
and information content, and there has been a transition from static data sets
to data streams that must be analyzed in real time. Interesting or anomalous
phenomena must be quickly characterized and followed up with additional
measurements via optimal deployment of limited assets. Modern astronomy
presents a variety of such phenomena in the form of transient events in digital
synoptic sky surveys, including cosmic explosions (supernovae, gamma ray
bursts), relativistic phenomena (black hole formation, jets), potentially
hazardous asteroids, etc. We have been developing a set of machine learning
tools to detect, classify and plan a response to transient events for astronomy
applications, using the Catalina Real-time Transient Survey (CRTS) as a
scientific and methodological testbed. The ability to respond rapidly to the
potentially most interesting events is a key bottleneck that limits the
scientific returns from the current and anticipated synoptic sky surveys.
Similar challenge arise in other contexts, from environmental monitoring using
sensor networks to autonomous spacecraft systems. Given the exponential growth
of data rates, and the time-critical response, we need a fully automated and
robust approach. We describe the results obtained to date, and the possible
future developments.},
Year          = {2014},
Month         = {Jul},
Url           = {http://arxiv.org/abs/1407.3502v1},
File          = {1407.3502v1.pdf}
}
```

</p></details>

[Download PDF](http://arxiv.org/pdf/1407.3502v1)

The nature of scientific and technological data collection is evolving rapidly: data volumes and rates grow exponentially, with increasing complexity and information content, and there has been a transition from static data sets to data streams that must be analyzed in real time. Interesting or anomalous phenomena must be quickly characterized and followed up with additional measurements via optimal deployment of limited assets. Modern astronomy presents a variety of such phenomena in the form of transient events in digital synoptic sky surveys, including cosmic explosions (supernovae, gamma ray bursts), relativistic phenomena (black hole formation, jets), potentially hazardous asteroids, etc. We have been developing a set of machine learning tools to detect, classify and plan a response to transient events for astronomy applications, using the Catalina Real-time Transient Survey (CRTS) as a scientific and methodological testbed. The ability to respond rapidly to the potentially most interesting events is a key bottleneck that limits the scientific returns from the current and anticipated synoptic sky surveys. Similar challenge arise in other contexts, from environmental monitoring using sensor networks to autonomous spacecraft systems. Given the exponential growth of data rates, and the time-critical response, we need a fully automated and robust approach. We describe the results obtained to date, and the possible future developments.

- [ ] Relevant
- [ ] Irrelevant

## 4. Real-Time Data Mining of Massive Data Streams from Synoptic Sky Surveys (2016)

<details>
<summary style="font-size:14px"><code>.bib</code></summary>
<p>

```bibtex
@article{1601.04385v1,
Author        = {S. G. Djorgovski and M. J. Graham and C. Donalek and A. A. Mahabal and A. J. Drake and M. Turmon and T. Fuchs},
Title         = {Real-Time Data Mining of Massive Data Streams from Synoptic Sky Surveys},
Eprint        = {1601.04385v1},
DOI           = {10.1016/j.future.2015.10.013},
ArchivePrefix = {arXiv},
PrimaryClass  = {astro-ph.IM},
Abstract      = {The nature of scientific and technological data collection is evolving
rapidly: data volumes and rates grow exponentially, with increasing complexity
and information content, and there has been a transition from static data sets
to data streams that must be analyzed in real time. Interesting or anomalous
phenomena must be quickly characterized and followed up with additional
measurements via optimal deployment of limited assets. Modern astronomy
presents a variety of such phenomena in the form of transient events in digital
synoptic sky surveys, including cosmic explosions (supernovae, gamma ray
bursts), relativistic phenomena (black hole formation, jets), potentially
hazardous asteroids, etc. We have been developing a set of machine learning
tools to detect, classify and plan a response to transient events for astronomy
applications, using the Catalina Real-time Transient Survey (CRTS) as a
scientific and methodological testbed. The ability to respond rapidly to the
potentially most interesting events is a key bottleneck that limits the
scientific returns from the current and anticipated synoptic sky surveys.
Similar challenge arise in other contexts, from environmental monitoring using
sensor networks to autonomous spacecraft systems. Given the exponential growth
of data rates, and the time-critical response, we need a fully automated and
robust approach. We describe the results obtained to date, and the possible
future developments.},
Year          = {2016},
Month         = {Jan},
Url           = {http://arxiv.org/abs/1601.04385v1},
File          = {1601.04385v1.pdf}
}
```

</p></details>

[Download PDF](http://arxiv.org/pdf/1601.04385v1)

The nature of scientific and technological data collection is evolving rapidly: data volumes and rates grow exponentially, with increasing complexity and information content, and there has been a transition from static data sets to data streams that must be analyzed in real time. Interesting or anomalous phenomena must be quickly characterized and followed up with additional measurements via optimal deployment of limited assets. Modern astronomy presents a variety of such phenomena in the form of transient events in digital synoptic sky surveys, including cosmic explosions (supernovae, gamma ray bursts), relativistic phenomena (black hole formation, jets), potentially hazardous asteroids, etc. We have been developing a set of machine learning tools to detect, classify and plan a response to transient events for astronomy applications, using the Catalina Real-time Transient Survey (CRTS) as a scientific and methodological testbed. The ability to respond rapidly to the potentially most interesting events is a key bottleneck that limits the scientific returns from the current and anticipated synoptic sky surveys. Similar challenge arise in other contexts, from environmental monitoring using sensor networks to autonomous spacecraft systems. Given the exponential growth of data rates, and the time-critical response, we need a fully automated and robust approach. We describe the results obtained to date, and the possible future developments.

- [ ] Relevant
- [ ] Irrelevant

## 5. Bennu's global surface and two candidate sample sites characterized by spectral clustering of OSIRIS-REx multispectral images (2021)

<details>
<summary style="font-size:14px"><code>.bib</code></summary>
<p>

```bibtex
@article{2104.02435v1,
Author        = {J. L. Rizos and J. de Leon and J. Licandro and D. R. Golish and H. Campins and E. Tatsumi and M. Popescu and D. N. DellaGiustina and M. Pajola and J. -Y. Li and K. J. Becker and D. S. Lauretta},
Title         = {Bennu's global surface and two candidate sample sites characterized by
  spectral clustering of OSIRIS-REx multispectral images},
Eprint        = {2104.02435v1},
DOI           = {10.1016/j.icarus.2021.114467},
ArchivePrefix = {arXiv},
PrimaryClass  = {astro-ph.EP},
Abstract      = {The OSIRIS-REx spacecraft encountered the asteroid (101955) Bennu on December
3, 2018, and has since acquired extensive data from the payload of scientific
instruments on board. In 2019, the OSIRIS-REx team selected primary and backup
sample collection sites, called Nightingale and Osprey, respectively. On
October 20, 2020, OSIRIS-REx successfully collected material from Nightingale.
In this work, we apply an unsupervised machine learning classification through
the K-Means algorithm to spectrophotometrically characterize the surface of
Bennu, and in particular Nightingale and Osprey. We first analyze a global
mosaic of Bennu, from which we find four clusters scattered across the surface,
reduced to three when we normalize the images at 550 nm. The three spectral
clusters are associated with boulders and show significant differences in
spectral slope and UV value. We do not see evidence of latitudinal
non-uniformity, which suggests that Bennu's surface is well-mixed. In our
higher-resolution analysis of the primary and backup sample sites, we find
three representative normalized clusters, confirming an inverse correlation
between reflectance and spectral slope (the darkest areas being the reddest
ones) and between b' normalized reflectance and slope. Nightingale and Osprey
are redder than the global surface of Bennu by more than $1\sigma$ from
average, consistent with previous findings, with Nightingale being the reddest
($S' = (- 0.3 \pm 1.0) \times 10^{- 3}$ percent per thousand angstroms). We see
hints of a weak absorption band at 550 nm at the candidate sample sites and
globally, which lends support to the proposed presence of magnetite on Bennu.},
Year          = {2021},
Month         = {Apr},
Url           = {http://arxiv.org/abs/2104.02435v1},
File          = {2104.02435v1.pdf}
}
```

</p></details>

[Download PDF](http://arxiv.org/pdf/2104.02435v1)

The OSIRIS-REx spacecraft encountered the asteroid (101955) Bennu on December 3, 2018, and has since acquired extensive data from the payload of scientific instruments on board. In 2019, the OSIRIS-REx team selected primary and backup sample collection sites, called Nightingale and Osprey, respectively. On October 20, 2020, OSIRIS-REx successfully collected material from Nightingale. In this work, we apply an unsupervised machine learning classification through the K-Means algorithm to spectrophotometrically characterize the surface of Bennu, and in particular Nightingale and Osprey. We first analyze a global mosaic of Bennu, from which we find four clusters scattered across the surface, reduced to three when we normalize the images at 550 nm. The three spectral clusters are associated with boulders and show significant differences in spectral slope and UV value. We do not see evidence of latitudinal non-uniformity, which suggests that Bennu's surface is well-mixed. In our higher-resolution analysis of the primary and backup sample sites, we find three representative normalized clusters, confirming an inverse correlation between reflectance and spectral slope (the darkest areas being the reddest ones) and between b' normalized reflectance and slope. Nightingale and Osprey are redder than the global surface of Bennu by more than $1\sigma$ from average, consistent with previous findings, with Nightingale being the reddest ($S' = (- 0.3 \pm 1.0) \times 10^{- 3}$ percent per thousand angstroms). We see hints of a weak absorption band at 550 nm at the candidate sample sites and globally, which lends support to the proposed presence of magnetite on Bennu.

- [ ] Relevant
- [ ] Irrelevant

