fn main() {
    let dst = cmake::Config::new(".")
        .build_target("reussir-allocator")
        .build();
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=static=reussir-allocator");
    // link c++ on apple
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=atomic");
    }

    if cfg!(target_os = "freebsd") {
        // using THREAD_DESTRUCTOR
    } else if cfg!(all(unix, not(target_os = "macos"))) {
        // using PTHREAD_DESTRUCTOR
        if cfg!(target_env = "gnu") {
            println!("cargo:rustc-link-lib=c_nonshared");
        }
    } else if cfg!(windows) {
        // not need for explicit c++ runtime
    } else {
        // link c++ runtime
        println!(
            "cargo:rustc-link-lib={}",
            std::env::var("CXXSTDLIB").unwrap_or_else(|_| {
                if cfg!(target_os = "macos") || cfg!(target_os = "openbsd") {
                    "c++".to_string()
                } else {
                    "stdc++".to_string()
                }
            })
        )
    }
}
