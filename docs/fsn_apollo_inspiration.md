# Lessons from Early 3D Interfaces and Apollo 11

This document highlights historical technology that informs the Knowledge3D vision.

## SGI IRIX fsn

[fsn](https://www.youtube.com/watch?v=k45Io79asvc) was a 3D file system navigator demoed on SGI workstations in the early 1990s. It presented directories as floating rooms with files represented as objects in space. The interface briefly appeared in the film *Jurassic Park* as a fanciful park control system.

Although fsn was largely a technology demo, it proved that spatial navigation of data can be intuitive and visually striking. Knowledge3D builds upon this spirit of exploration—rendering knowledge graphs as navigable 3D scenes so users "fly" through information just as fsn let users fly through directories.

## Apollo 11 modular software

NASA's Apollo Guidance Computer software was split into many small modules, each responsible for a distinct function such as navigation or user interface. This modular approach allowed independent development and rigorous testing of every component. When problems occurred during the mission, engineers could isolate and patch individual modules without rewriting the entire system.

Knowledge3D adopts a similar philosophy: data ingestion, processing, and visualization components are designed as loosely coupled modules. This makes the system easier to extend—new data formats or visualization techniques can slot in without breaking the rest of the pipeline.
