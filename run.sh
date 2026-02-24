#!/bin/bash

# ssh-add -k ~/.ssh/id_rsa_laptop

clear

git pull
git add .
git commit -m "update"
git push
