#!/usr/bin/env cwl-runner

cwlVersion: v1.2
class: CommandLineTool
baseCommand: /bil/data/hackathon/2022_GYBS/output/fMOST/noisysky/brainreg.sif
requirements:
  EnvVarRequirement:
    envDef:
      SINGULARITY_BINDPATH: "/bil/data/hackathon/2022_GYBS/input/fMOST/subject,/bil/data/hackathon/2022_GYBS/output/fMOST/noisysky"
inputs:
  message:
    type: string
    inputBinding:
      position: 1
outputs: []
