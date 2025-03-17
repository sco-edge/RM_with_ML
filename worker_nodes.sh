#!/usr/bin/env bash

# config for worker nodes only
kubeadm join --token 123456.1234567890123456 \
             --discovery-token-unsafe-skip-ca-verification $1:6443

# for Erms worker_nodes init
echo vagrant | sudo -S apt install unzip
git init
git config core.sparseCheckout true
git remote add -f origin https://github.com/Cloud-and-Distributed-Systems/Erms.git
echo "additionalFiles/">> .git/info/sparse-checkout
git pull origin main
cd additionalFiles
unzip media.zip
unzip social.zip

