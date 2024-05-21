#! /bin/bash

while true
do
    PORT=$((RANDOM%1000+12003))
    echo $PORT
    status="$(nc -z 127.0.0.1 $PORT; echo $?)"
    echo $status

    if [ "$status" -eq "1" ]; then
        break;
    fi
done