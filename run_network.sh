#!/bin/bash

docker run -it --rm -p 4567:4567 -v `pwd`:`pwd` -w `pwd` udacity/carnd-term1-starter-kit python drive.py model.h5
