#!/bin/bash

TC=/sbin/tc

# interface traffic will leave on
IF=wlan0

# run clean to ensure existing tc is not configured
clean () {
  echo "== CLEAN INIT =="
  $TC qdisc del dev $IF root
  echo "== CLEAN DONE =="
}

clean

