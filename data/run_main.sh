#!/bin/bash

for i in `seq 0 49`;
    do
    	qsub -VRUN_NUMBER=i run_parser.sh
    done