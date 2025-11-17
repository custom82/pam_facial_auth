# pam_facial_auth

A pluggable authentication module that relies on facial recognition to
verify the user attempting to gain access. Looks for a match between
username and label associated with the recognized face.

Requirements
------------
- OpenCV 4 including contrib module
- PAM devel

Quickstart
----------
Clone this repository
```
git clone https://github.com/custom82/pam_facial_auth.git
```
Build and install facial auth module
------------------------------------

- cd pam_facial_auth
- mkdir build
- cmake ..
- make
- make install

How it Work
-----------

- Configure /etc/security/pam_facial.conf with opencv haar xml and other paramenter

- Capture Face Images

facial_capture -u user -d /dev/video0 -g -v

- Train your images

facial_training -u user -m lpbh -v 

- Test Authentication

facial_test -u user -n -v 


- Add pam_facial_auth to your pam stack

example 
auth pam_facial_auth.so debug=true nogui=true


- Legal Information
pam_facial_auth is released under the GNU GPL 3
