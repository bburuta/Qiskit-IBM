#!/bin/bash

sinfo --partition=QPU --Node \
    --Format="NodeList:12,StateLong:12,CPUsState:18,CPUs:6,Memory:10,Gres:60,GresUsed:60"
