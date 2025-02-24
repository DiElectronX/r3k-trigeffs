#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, field
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import *
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.utils.crabhelper import inputFiles, runsAndLumis


@dataclass
class ParamSet:
    outDir: str = '.'
    inputFiles: list[str] = field(default_factory=lambda: inputFiles())
    cut: str = 'nElectron > 1 && Electron_pt[0] > 5. && Electron_pt[1] > 5. && abs(Electron_eta[0]) < 1.22 && abs(Electron_eta[1]) < 1.22 && Electron_charge[0] + Electron_charge[1] == 0 && (HLT_DoubleMu4_3_Bs || HLT_DoubleMu4_3_Jpsi || HLT_DoubleMu4_3_LowMass || HLT_DoubleMu4_LowMass_Displaced || HLT_Mu0_L1DoubleMu || HLT_Mu4_L1DoubleMu || HLT_DoubleMu3_Trk_Tau3mu || HLT_DoubleMu3_TkMu_DsTau3Mu || HLT_DoubleMu4_MuMuTrk_Displaced || HLT_DoubleMu4_Jpsi_Displaced || HLT_DoubleMu4_Jpsi_NoVertexing || HLT_DoubleMu4_JpsiTrkTrk_Displaced || HLT_DoubleMu4_JpsiTrk_Bc || HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass || HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05)'
    json: str = 'Eras_CDEFG/trigger_OR.json'
    branchesIn: str = None
    branchesOut: str = None
    label: str = '_skim'
    friend: bool = False
    maxEntries: int = None
    haddFileName: str = 'output_skim.root'


class SkimEvents(Module):
    def __init__(self):
        self.writeHistFile = True
    def beginJob(self, histFile=None, histDirName=None):
        Module.beginJob(self, histFile, histDirName)
    def analyze(self, event):
        return True


params = ParamSet()
p = PostProcessor(
        params.outDir,
        params.inputFiles,
        cut=params.cut,
        postfix=params.label,
        branchsel=params.branchesIn,
        outputbranchsel=params.branchesOut,
        modules=[SkimEvents()],
        haddFileName=params.haddFileName,
        compression=('LZMA:9'),
        friend=params.friend,
        maxEntries=params.maxEntries,
        jsonInput=params.json,
        provenance=True,
        fwkJobReport=True,
)
p.run()

print("DONE")
