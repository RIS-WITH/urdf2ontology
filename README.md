# URDF2Ontology

While symbolic reasoning over task allocation requires grounding a robot's physical capabilities, standard Universal Robot Description Format (URDF) files provide low-level kinematics without formal semantic context to be reasoned on. We present URDF2Ontology, an automated, taxonomy-agnostic neurosymbolic framework that transforms raw URDF into populated, formally grounded OWL ontologies. To eliminate LLM context window saturation during graph parsing, our system introduces an iterative sequential graph-purging mechanism paired with a morphology-aware dual-track architecture: a bottom-up Tree Track for branching assemblies and a top-down Chain Track for continuous serial limbs. Using constrained LLM primitives, URDF2Ontology extracts functional sub-graphs and asserts formally aligned populated ontology against arbitrary user-provided taxonomy.

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
