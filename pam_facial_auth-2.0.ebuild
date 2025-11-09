EAPI=8

DESCRIPTION="PAM facial authentication module"
HOMEPAGE="https://github.com/devinaconley/pam-facial-auth/"
SRC_URI="https://github.com/devinaconley/pam-facial-auth/commit/1184b9c961959b4bf3b74a0e7f7c22c9541927d7.tar.gz -> ${P}.tar.gz"

LICENSE="GPL-3"
SLOT="0"
KEYWORDS="~amd64"
IUSE=""

DEPEND="dev-libs/opencl
        media-libs/opencv
        sys-libs/pam"
RDEPEND="${DEPEND}"




src_prepare() {
    default
    # Aggiungi eventuali patch, se necessario
}

src_configure() {
    # Configura il progetto (cmake)
    cmake-utils_src_configure
}

src_compile() {
    # Compila il progetto
    cmake-utils_src_compile
}

src_install() {
    # Installa i binari
    cmake-utils_src_install

    # Installa la man page
    insinto /usr/share/man/man8
    doman "${S}/man/pam-facial-auth.8"  # Aggiungi il percorso corretto della man page

    insinto /etc/pam-facial-auth/
    doins ${S}/etc/haarcascade_frontalface_default.xml
    doins ${S/etc/haarcascade_frontalface_alt.xml}


}

