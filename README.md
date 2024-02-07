# Personalising Crutches

### Abstract 
Crutches are optimised for stable motion, but this safety comes at the cost of comfort and speed. In this paper, I employ Gaussian Processes (GPs) and Bayesian Optimisation (BO) as hypothesis generators to find better crutch configurations, which I validate on a physical prototype. I do so by defining a novel loss function indicating the quality of a crutch design which combines subjective metrics (joint pain, instability and effort) and the corresponding objective ones. Finally, I (1) use this methodology to build a more stable, less effortful and less painful personalised crutch design and (2) use the knowledge built by the GP through these experiments to enhance the understanding of the physical dynamics of crutching.


### Data & analysis
- Accelerometer data collected through Polar H10 monitor, extracted via FingerPulseLatency software built by Christof Schwiening (https://github.com/cjs30/FingerPulseLatency). 
- Steps detected via futher software from Christof Schwiening, creating one .csv file with accelerometer data and one .step file with the time for each step and amplitude of the X+Z accelerometer at each step.
- Further analysis requires .csv (accelerometer) and .step files to extract the various loss metrics indicative of effort, instability, and pain.
- These are then handed to the GP BO which will then suggest a novel better geometry. This is tested through a simulator, built and run on the user.


 



