#!/bin/bash
cd /vol/research/lyc/OpenIE-standalone/
/vol/research/lyc/jdk1.8.0_202/bin/java -Xmx10g -XX:+UseConcMarkSweepGC -jar /vol/research/lyc/OpenIE-standalone/openie-assembly-5.0-SNAPSHOT.jar --httpPort 8000 &
P1=$!
/user/HS502/yl02706/.conda/envs/lyc/bin/python /vol/research/lyc/EmpatheticMeta/data/opinion_mining.py $1 /vol/research/lyc/EmpatheticMeta/data/opinion_openie &&
kill $P1