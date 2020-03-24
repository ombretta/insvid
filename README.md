# instructional_videos_project

Version 0.1.0

This repository contains the code to train a simple Deep Learning model to predict what task is being performed in an instructional video.

The model consists in a simple fully connected layer that takes as input precomputed i3d features (Carreira, Joao, and Andrew Zisserman. "Quo vadis, action recognition? a new model and the kinetics dataset." proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.).

This repository contains the code to train the model on the CrossTask dataset (Zhukov, Dimitri, et al. "Cross-task weakly supervised learning from instructional videos." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.). 

The i3d features are not contained in the repository and you need to precompute them first. 

Requirements: 
- Python 3.6.0
- torch 1.4.0
- CrossTask (https://github.com/DmZhukov/CrossTask) 


### Source code
You can find the code to train the model under ./src/.
Run python simple_classifier.py to load the CrossTask dataset and train the fully connected layer. 

That will generate results in the ./data/ folder. You can evaluate the results running:
python analyse_results.py


## Project organization

```
.
├── .gitignore
├── CITATION.md
├── LICENSE.md
├── README.md
├── requirements.txt
├── bin                <- Compiled and external code, ignored by git (PG)
│   └── external       <- Any external source code, ignored by git (RO)
├── config             <- Configuration files (HW)
├── data               <- All project data, ignored by git
│   ├── processed      <- The final, canonical data sets for modeling. (PG)
│   ├── raw            <- The original, immutable data dump. (RO)
│   └── temp           <- Intermediate data that has been transformed. (PG)
├── docs               <- Documentation notebook for users (HW)
│   ├── manuscript     <- Manuscript source, e.g., LaTeX, Markdown, etc. (HW)
│   └── reports        <- Other project reports and notebooks (e.g. Jupyter, .Rmd) (HW)
├── results
│   ├── figures        <- Figures for the manuscript or reports (PG)
│   └── output         <- Other output for the manuscript or reports (PG)
└── src                <- Source code for this project (HW)

```


## License

This project is licensed under the terms of the [MIT License](/LICENSE.md)

## Citation

Please [cite this project as described here](/CITATION.md).
