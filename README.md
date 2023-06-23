This repository is the python implementation of 1-body part in paper https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.224309  

1. Eigenvalue problem for pbc
(a) To compute eigenvalue problem for 1 ratio a/b, run cns.py
(b) To scan different ratios a/b, run runTwoPairCNS.py, where twoPairCNS.py is imported.
(c) To compute eigenvalue problem for omegaF=0, run cnsOmegaF0.py

2. To compute pumping
(a) For Wannier initial state, run cnsPumping.py
(b) For Gaussian initial state, run gaussianCnsPumping.py

3. To compute obc bands
(a) For some a/b, run obc.py
(b) For omegaF=0, run obcOmegaF0.py

4. To plot pumping results for 3 pbc bands, run plt3Pumpings.py

5. To plot pumping results for 3 pbc bands with omegaF=0, run plt3PumpingsOmegaF0.py

6. To plot obc bands, run pltObc.py

7. To plot obc bands for omegaF=0, run pltObcOmegaF0.py
