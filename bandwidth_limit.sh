#!/bin/bash

TC=/sbin/tc

# interface traffic will leave on
IF=wlan0

# The parent limit, children can borrow from this amount of bandwidth
# based on what's available.
LIMIT=20mbit

# host 1
DST_CIDR=192.168.1.38/32

# filter command -- add ip dst match at the end
U32="$TC filter add dev $IF protocol ip parent 1:0 prio 1 u32"

create () {
  echo "== SHAPING INIT =="

  # create the root qdisc
  $TC qdisc add dev $IF root handle 1:0 htb \
    default 30

  # create the parent qdisc, children will borrow bandwidth from
  $TC class add dev $IF parent 1:0 classid \
    1:1 htb rate $LIMIT

  # setup filters to ensure packets are enqueued to the correct
  $U32 match ip dst $DST_CIDR flowid 1:1

  echo "== SHAPING DONE =="
}

# run clean to ensure existing tc is not configured
clean () {
  echo "== CLEAN INIT =="
  $TC qdisc del dev $IF root
  echo "== CLEAN DONE =="
}

clean
create
