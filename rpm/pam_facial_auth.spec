Name:           pam-facial-auth
Version:        1.0
Release:        1%{?dist}
Summary:        PAM module for facial recognition authentication

License:        GPLv3+
URL:            https://example.com/
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  cmake, gcc-c++, pam-devel, opencv-devel
Requires:       pam

%description
This package contains a PAM module that performs facial authentication
using OpenCV and LBPH algorithms.

%prep
%setup -q

%build
%cmake .
%cmake_build

%install
%cmake_install

# Create directories
mkdir -p %{buildroot}/etc/pam_facial_auth/images
mkdir -p %{buildroot}/etc/pam_facial_auth/models

%files
%license LICENSE
/usr/lib64/security/pam_facial_auth.so
/usr/lib64/libfacialauth.so
/usr/sbin/facial_capture
/usr/sbin/facial_training
/usr/sbin/facial_test
/usr/share/man/man1/facial_capture.1.gz
/usr/share/man/man1/facial_training.1.gz
/usr/share/man/man1/facial_test.1.gz
/usr/share/man/man8/pam_facial_auth.8.gz
/etc/security/pam_facial.conf
/etc/pam_facial_auth/

%changelog
* Tue Nov 18 2025 Andrea Postiglione 1.0-1
- Initial RPM release
