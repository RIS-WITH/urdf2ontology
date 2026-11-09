# URDF2Ontology

While symbolic reasoning for task allocation requires grounding a robot’s physical capabilities, standard Unified Robot Description Format (URDF) files provide only low-level kinematics without the formal semantic context needed for reasoning. We present URDF2Ontology, an automated, taxonomy-agnostic neurosymbolic framework that transforms raw URDF files into populated, grounded OWL ontologies. It leverages a Large Language Model (LLM) for targeted semantic reasoning, coupled with an iterative sequential graph-purging mechanism to prevent context window saturation. To process complex kinematic structures, it employs a morphology-aware dual-track architecture featuring a bottom-up Tree Track for branching assemblies and a top-down Chain Track for serial limbs. Guided by neurosymbolic guardrails, the framework extracts functional subgraphs and asserts populated ontologies aligned with arbitrary user-provided taxonomies. Empirical evaluation across a diverse fleet of heterogeneous robots demonstrates strong schema alignment, valid structural extraction, and a significant reduction in manual domain authoring overhead.

# Project Structure
```
├── urdf2ontology/
│   ├── urdf_models/     # Source URDF files
│   ├── new_evaluation/  # Results with logs and OWL files
│   ├── scripts/         # Scripts to run the abstraction process
│       ├── tasks/       # folder containing each of the python file for the tasks
│       └── main.py      # main python file to run the abstraction
│   └── ontology/       # OWL/TTL files for the domain-expert ontology
└── README.md
```

# Quick Start (Local Setup)
```
1. pip install ollama (or create a virtual environment with ollama)
2. install ontologenius from the dedicated github repository ([link](https://github.com/sarthou/ontologenius))
3. Load the URDF files in urdf2ontology/urdf2ontology/urdf_models/ folder
4. Add their paths to the main.py
5. Run ontologenius with ros2 launch ontologenius ontologeniu_empty.xml
6. Run the main loop with python3 main.py
```
## Citation
If you use this work in your research, please cite the following paper as below:

BibTeX:
```bibtex
@article{dussard2026taxonomy,
  title={Taxonomy-Agnostic Neurosymbolic Grounding of Robot Kinematics},
  author={Dussard, Bastien and Clodic, Aurélie and Sarthou, Guillaume},
  journal={Proceedings of the AAAI Symposium Series},
  url={https://ojs.aaai.org/index.php/AAAI-SS/article/view/43504},
  DOI={10.1609/aaaiss.v10i1.43504},
  number={1},
  volume={10},
  year={2026},
  month={Nov.},
  pages={895–903}
}
