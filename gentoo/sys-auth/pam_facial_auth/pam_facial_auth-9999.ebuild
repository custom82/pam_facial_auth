# Copyright 1999-2025 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2

EAPI=8

inherit cmake

COMMIT="42701d24a05057735575277d669af836cbcee22e"
DESCRIPTION="PAM facial authentication module"
HOMEPAGE="https://github.com/custom82/pam_facial_auth/"

if [[ ${PV} == 9999 ]] ; then
        inherit git-r3
        EGIT_REPO_URI="https://github.com/custom82/pam_facial_auth.git"
else
        SRC_URI="https://github.com/custom82/pam_facial_auth/archive/${COMMIT}.tar.gz: > ${P}.tar.gz"
        KEYWORDS="~amd64"
		S="${PN}-${COMMIT}"
fi


LICENSE="GPL-3"
SLOT="0"
KEYWORDS="~amd64"
IUSE=""

DEPEND="
		media-libs/opencv[contrib]
		sys-libs/pam
"


RDEPEND="${DEPEND}"

src_prepare() {
		cmake_src_prepare
}

src_configure() {
		cmake_src_configure
}

src_compile() {
		cmake_src_compile
}

src_install() {
		cmake_src_install

}

