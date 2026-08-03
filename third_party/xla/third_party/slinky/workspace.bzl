"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "0bc38e01bdab4743104e94a6fe9e20f79a1792fe1297a4af7a23bfb679d85486",
        strip_prefix = "slinky-c025c1f0e35c1828363f74affcd7852f57cfdf1b",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/c025c1f0e35c1828363f74affcd7852f57cfdf1b.zip"),
    )
