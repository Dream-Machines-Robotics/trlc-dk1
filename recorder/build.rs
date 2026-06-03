fn main() {
    // Link the system libturbojpeg (libturbojpeg0-dev). Avoids the turbojpeg
    // crate's from-source libjpeg-turbo build (needs cmake+nasm, absent here).
    println!("cargo:rustc-link-lib=turbojpeg");
}
