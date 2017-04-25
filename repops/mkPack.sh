#!/bin/bash
rm tprod.zip repop.zip
echo tprod
cd tprod; zip ../tprod.zip *.m *.c *.h *.def readme makefile 
echo repop
cd ../repop; zip ../repop.zip *.m *.c *.h *.def readme makefile 
