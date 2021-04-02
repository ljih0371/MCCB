#!/bin/sh
LINE_LIST="ABH125c_1 ABH250c_1 ABH125c_2 ABH250c_2"

for line in $LINE_LIST
do
  find /data/mccb/$line -name '*.png' -mtime +3 -delete
  find /data/mccb/$line -name '*.xml' -mtime +3 -delete
done

