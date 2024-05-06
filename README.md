# Calculating and Plotting Trigger Efficiencies for Run 3 R(K)

## Setting Up

### Building Environment (First-Time Setup)
```
cmssw-el7
cmsrel CMSSW_14_0_6
cd CMSSW_14_0_6/src
cmsenv
git clone git@github.com:DiElectronX/r3k-trigeffs.git
cd $CMSSW_BASE/src/r3k-trigeffs
```

### Loading Environment (When Logging In)
```
cd .../r3k-trigeffs
cmsenv
```

## Orthogonal Dataset Method (Double-Muon Trigger) [WIP porting code to GIT]

### Skimming NanoAOD files

For ease of use, loose pre-selection cuts are applied to slim data size and allow interactive production of trigger efficiency results.

```
python skim_nanoaod.py
```

### Calculating Efficiencies

The code builds histograms containing the events in the numerator and denominator of relevant efficiency plots.

```
python calculate_efficiencies.py
```

### Plotting Efficiencies

Code is broken now but it shows the process of how the efficiency plots are calculated. Will fix soon.


This code reads the histograms from the previous step, performs a simplified fit of the J/Psi peak to obtain a signal yield, and uses the yields to generate trigger efficiency plots.

```
python calculate_efficiencies.py
```


