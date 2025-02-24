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
    cut: str = {CUT_TEMPLATE}
    json: str = {JSON_TEMPLATE}
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
