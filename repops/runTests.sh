#!/bin/bash -x
# convert from etprod log output into a rand-init tprod_testcase call

calllog=$1
if [ -z $calllog ] ; then calllog=/tmp/tprod.log; fi

cat $calllog | sort | uniq > $calllog.uniq

sed  -e 's/\(\[[^]]*\]\) \(([^)]*)\)/\2 \1 rand/g' -e 's/(double)/dr/g' -e 's/(double complex)/dc/g' -e 's/(single)/sr/g' -e 's/(single complex)/sc/g' -e 's/,/ /g' -e's/[()]/ /g' -e 's/\[/"\[/g' -e 's/\]/\]"/g' $calllog.uniq > $calllog.testcases

sed -e 's#\(tprod\)#valgrind ~/source/matfiles/repops/\1/\1_testcases#g' -e 's#\(repop\)#valgrind ~/source/matfiles/repops/\1/\1_testcases#g' -e 's#$# > /dev/null#g' $calllog.testcases > $calllog.final

bash -x $calllog.final 2>&1

