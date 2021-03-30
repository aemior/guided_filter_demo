# Guided Filter Demo

## Usage

```shell
$mkdir build
$cd build
$cmake ..
$make
$./guided_filter_process ../imgs/cave-flash.png ../imgs/cave-noflash.png 8 0.0004 2 out.png out_fast.png
```

## TODO

- radius is generator parameter not input parameter.
- scale factor is fix to 2.
- simple schedule,just some compute_root and parallel. 