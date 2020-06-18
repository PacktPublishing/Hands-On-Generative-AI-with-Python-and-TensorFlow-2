#!/bin/bash

export DOWNLOAD_LOCATION=~/Downloads/
export VAGRANT_URL=https://releases.hashicorp.com/vagrant/2.2.6/vagrant_2.2.6_x86_64.dmg
export VIRTUALBOX_URL=https://download.virtualbox.org/virtualbox/6.1.2/VirtualBox-6.1.2-135662-OSX.dmg

echo "===== Install Vagrant ====="

wget $VAGRANT_URL -P $DOWNLOAD_LOCATION
hdiutil attach $DOWNLOAD_LOCATION"vagrant_2.2.6_x86_64.dmg"
sudo installer -pkg /Volumes/Vagrant/vagrant.pkg -target /

echo "===== Install VirtualBox ====="

wget $VIRTUALBOX_URL -P $DOWNLOAD_LOCATION
hdiutil attach $DOWNLOAD_LOCATION"VirtualBox-6.1.2-135662-OSX.dmg"
sudo installer -pkg /Volumes/VirtualBox/virtualbox.pkg -target /

echo "===== Install MiniKF ====="
vagrant init arrikto/minikf
vagrant up