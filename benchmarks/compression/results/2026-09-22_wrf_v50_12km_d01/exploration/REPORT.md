> **Archived first-round exploration (2026-09-22).** Kept as the record of what was tried. Where it disagrees with
> the Findings in `benchmarks/compression/README.md` — which were revised after review round `cfdb-compression-1`
> and extended with lz4 and pure-numpy filters — the README is current.

# cfdb chunk compression assessment — 2026-09-22

Data: `~/data/wrf/sst/v50_12km_wvt_8_regions/cfdb_cache/d01.cfdb` (557 MB, zstd-1, 33 data vars,
231 chunks of (24,1,324,277); 27 vars uint16-packed, 3 uint32-packed, 2 uint8, 1 raw float32).
Every codec was fed the exact encoded array cfdb hands its compressor, and every chunk was
round-tripped and asserted bit-identical. Single thread (OMP_NUM_THREADS=1, blosc nthreads=1)
unless labelled. CPU: Ryzen 9 7900. Throughput = raw (uncompressed) MB/s, min of 2 runs.

"HARD" = the 23 vars where zstd-1 gets < 5x (550 of the 556 MB). The other 10 (masks, snow,
sea-ice, terrain, albedo, emissivity, RH) compress to ~nothing under everything.

## All codecs, HARD subset, sorted by ratio
codec                                       ratio  size MB vs zstd1  comp MB/s  decomp MB/s | ALL ratio  file MB
pcodec L12                                   2.84    296.9     0.54         45         1092 |      3.58    311.8
pcodec L8 (default)                          2.83    298.0     0.54        244         1235 |      3.56    313.2
pcodec L8 delta1                             2.66    316.9     0.58        288         1327 |      3.36    332.3
pcodec L8 lookback                           2.66    317.3     0.58         74          592 |      3.35    332.8
blosc2 zstd-9 shuffle+bytedelta              2.60    325.4     0.59         10         1931 |      3.38    330.5
pcodec L4                                    2.59    325.9     0.59        299         1245 |      3.24    344.3
blosc2 zstd-5 shuffle+bytedelta              2.52    334.9     0.61        127         2213 |      3.18    350.4
blosc2 zstd-3 shuffle+bytedelta              2.48    340.5     0.62        295         1817 |      3.13    356.7
blosc2 zstd-1 shuffle+bytedelta              2.47    342.1     0.62       1026         2010 |      3.11    358.4
blosc2 zstd-1 shuffle+bytedelta 12thr        2.47    342.1     0.62       4380         6319 |      3.11    358.4
npdelta+npshuffle+zstd-3                     2.25    375.4     0.68        506          492 |      2.93    381.0
npdelta+npshuffle+zstd-1                     2.23    377.9     0.69        705          491 |      2.91    383.7
npshuffle+zstd-3                             2.17    389.8     0.71        860          817 |      2.83    394.5
blosc2 zstd-1 shuffle                        2.14    394.9     0.72       1287         3353 |      2.69    414.7
npshuffle+zstd-1                             2.13    396.0     0.72       1155          835 |      2.79    400.6
blosc1 zstd-1 shuffle (python-blosc)         2.08    406.8     0.74        977         3420 |         -        -
blosc2 zstd-1 bitshuffle                     2.07    408.3     0.74       1210         1913 |      2.60    429.6
blosc1 zstd-1 bitshuffle (python-blosc)      2.07    408.3     0.74       1054         1925 |         -        -
bitshuffle+zstd-1                            2.02    417.3     0.76        534         1349 |      2.53    440.2
pcodec L2                                    1.98    427.2     0.78        361         1284 |      2.46    453.8
blosc2 zstd-1 bitshuffle+bytedelta           1.96    430.9     0.78       1212         1794 |      2.45    454.7
bitshuffle+lz4                               1.87    452.1     0.82       2008         3998 |      2.32    481.7
blosc2 lz4 shuffle+bytedelta                 1.87    452.5     0.82       2101         4109 |      2.33    478.3
blosc2 lz4 shuffle                           1.85    456.4     0.83       2406         6038 |      2.32    481.9
blosc2 blosclz shuffle                       1.83    460.3     0.84        942         4221 |      2.30    485.2
blosc2 lz4 bitshuffle                        1.83    460.7     0.84       2679         4597 |      2.27    491.2
zfp lossless                                 1.82    464.6     0.84        172          386 |      2.25    495.8
zstd-9                                       1.81    465.4     0.85         97         1421 |      2.37    470.3
zstd-6                                       1.77    475.7     0.86        138         1431 |      2.32    480.7
zstd-3                                       1.70    497.9     0.91        291         1365 |      2.22    503.3
fpzip lossless (f32 only)                    1.61     37.5     0.07        142          152 |      1.61     37.5  (ivt only)
zstd-1 (current)                             1.53    550.1     1.00        696         1796 |      2.01    556.3
lz4-1                                        1.19    710.7     1.29       1198         7179 |      1.49    746.3

## Per-chunk wall clock, one 4.3 MB air_temperature chunk (context: cfdb's own decode = 1.0 ms, encode = 1.3 ms)
zstd-1 (current)                           compress    6.0 ms   decompress   2.6 ms   -> 3.14 MB
zstd-3                                     compress   19.6 ms   decompress   4.0 ms   -> 2.60 MB
blosc2 zstd-1 shuffle                      compress    2.4 ms   decompress   0.9 ms   -> 2.28 MB
blosc2 zstd-1 shuffle+bytedelta            compress    3.6 ms   decompress   2.1 ms   -> 1.68 MB
blosc2 zstd-1 shuffle+bytedelta 12thr      compress    0.9 ms   decompress   0.7 ms   -> 1.68 MB
pcodec L4                                  compress   17.4 ms   decompress   4.3 ms   -> 1.68 MB
pcodec L8 (default)                        compress   20.0 ms   decompress   4.3 ms   -> 1.52 MB

## Chunk-size sensitivity (air_temperature, surface_pressure, precipitation, ivt)

### block 4.3MB (24,1,324,277)
codec                                 ratio  comp MB/s  decomp MB/s  us/call comp  us/call dec
zstd-1 (current)                       1.51        665         1804        9721.0       3582.5
lz4-1                                  1.19       1185         6036        5454.8       1070.5
blosc2 zstd-1 shuffle                  2.04       1396         3641        4627.3       1774.5
blosc2 zstd-1 shuffle+bytedelta        2.32       1144         2406        5650.1       2685.4
blosc2 lz4 shuffle                     1.85       2673         6937        2417.7        931.5
pcodec L4                              2.37        350         1549       18483.5       4172.5
pcodec L8 (default)                    2.56        276         1518       23399.4       4256.0

### block 180KB (1,1,324,277)
codec                                 ratio  comp MB/s  decomp MB/s  us/call comp  us/call dec
zstd-1 (current)                       1.50        683         1769         394.0        152.2
lz4-1                                  1.19       1288         6154         209.1         43.7
blosc2 zstd-1 shuffle                  2.03       1449         3954         185.9         68.1
blosc2 zstd-1 shuffle+bytedelta        2.29       1161         2262         231.8        119.0
blosc2 lz4 shuffle                     1.85       2928         7113          92.0         37.9
pcodec L4                              2.38        346         1522         778.1        176.9
pcodec L8 (default)                    2.56        240         1482        1120.6        181.7

### block 49KB (24,1,32,32)
codec                                 ratio  comp MB/s  decomp MB/s  us/call comp  us/call dec
zstd-1 (current)                       1.57        673         1651         109.5         44.7
lz4-1                                  1.24       1090         4493          67.6         16.4
blosc2 zstd-1 shuffle                  2.05       1423         3189          51.8         23.1
blosc2 zstd-1 shuffle+bytedelta        2.33       1178         2178          62.6         33.9
blosc2 lz4 shuffle                     1.84       2979         6234          24.7         11.8
pcodec L4                              2.39        329         1462         224.3         50.4
pcodec L8 (default)                    2.49        159         1411         462.3         52.3

### block 2KB (1,1,32,32)
codec                                 ratio  comp MB/s  decomp MB/s  us/call comp  us/call dec
zstd-1 (current)                       1.52        392          571           7.8          5.4
lz4-1                                  1.13       1051         1075           2.9          2.9
blosc2 zstd-1 shuffle                  1.95        356          576           8.6          5.3
blosc2 zstd-1 shuffle+bytedelta        2.12        289          461          10.6          6.7
blosc2 lz4 shuffle                     1.75        780          826           3.9          3.7
pcodec L4                              2.39         88          620          35.1          5.0
pcodec L8 (default)                    2.43         31          524          98.2          5.9

## Side question: float32 direct vs cfdb's uint16 packing (bytes per element; lower is better)
var                pipeline                                         bytes/elem comp MB/s(raw f32)  decomp MB/s
air_temperature    uint16 packed + zstd-1 (current)                      1.459               1059         3194
air_temperature    uint16 packed + blosc2 zstd-1 shuffle+bytedelta       0.779               2272         3943
air_temperature    uint16 packed + pcodec L8 (default)                   0.697                410         1996
air_temperature    float32 direct + zstd-1 (current)                     1.891                381         1406
air_temperature    float32 direct + blosc2 zstd-1 shuffle+bytedelta      1.500               1417         2352
air_temperature    float32 direct + pcodec L8 (default)                  0.915                354         2152
air_temperature    float32 direct + zfp lossless                         2.172                355          466
air_temperature    float32 direct + fpzip lossless (f32 only)            1.596                160          179

surface_pressure   uint16 packed + zstd-1 (current)                      2.340                428         1547
surface_pressure   uint16 packed + blosc2 zstd-1 shuffle+bytedelta       1.219               1364         2489
surface_pressure   uint16 packed + pcodec L8 (default)                   0.937                354         1958
surface_pressure   float32 direct + zstd-1 (current)                     2.587                603         1987
surface_pressure   float32 direct + blosc2 zstd-1 shuffle+bytedelta      1.194                987         2265
surface_pressure   float32 direct + pcodec L8 (default)                  1.228                294         1967
surface_pressure   float32 direct + zfp lossless                         1.878                350          515
surface_pressure   float32 direct + fpzip lossless (f32 only)            1.317                170          176

wind_speed         uint16 packed + zstd-1 (current)                      1.536               2354         3703
wind_speed         uint16 packed + blosc2 zstd-1 shuffle+bytedelta       0.978               1742         2954
wind_speed         uint16 packed + pcodec L8 (default)                   0.824                395         2004
wind_speed         float32 direct + zstd-1 (current)                     2.659                520         1698
wind_speed         float32 direct + blosc2 zstd-1 shuffle+bytedelta      2.123               1048         2085
wind_speed         float32 direct + pcodec L8 (default)                  0.962                219         1064
wind_speed         float32 direct + zfp lossless                         3.000                340          362
wind_speed         float32 direct + fpzip lossless (f32 only)            2.522                137          150

mixing_ratio       uint16 packed + zstd-1 (current)                      1.812               2507         3762
mixing_ratio       uint16 packed + blosc2 zstd-1 shuffle+bytedelta       1.247               1517         3581
mixing_ratio       uint16 packed + pcodec L8 (default)                   1.082                340         2006
mixing_ratio       float32 direct + zstd-1 (current)                     3.391               1121         2190
mixing_ratio       float32 direct + blosc2 zstd-1 shuffle+bytedelta      2.342                952         2381
mixing_ratio       float32 direct + pcodec L8 (default)                  2.180                271         1978
mixing_ratio       float32 direct + zfp lossless                         2.745                344          386
mixing_ratio       float32 direct + fpzip lossless (f32 only)            2.273                148          164

pwat               uint16 packed + zstd-1 (current)                      1.101                950         3052
pwat               uint16 packed + blosc2 zstd-1 shuffle+bytedelta       0.583               1646         4237
pwat               uint16 packed + pcodec L8 (default)                   0.482                479         2008
pwat               float32 direct + zstd-1 (current)                     1.288                444         1606
pwat               float32 direct + blosc2 zstd-1 shuffle+bytedelta      1.285                556         1701
pwat               float32 direct + pcodec L8 (default)                  0.586                253         1161
pwat               float32 direct + zfp lossless                         2.876                346          382
pwat               float32 direct + fpzip lossless (f32 only)            2.059                151          166


## Per-variable detail
=== per-variable compression ratio (hard vars) ===
var                                dtype    zstd-1 (curren         zstd-3 npshuffle+zstd blosc2 zstd-1  blosc2 zstd-1  blosc2 lz4 shu      pcodec L4 pcodec L8 (def   zfp lossless
air_temperature                    uint16             1.37           1.66           1.88           1.88           2.53           1.74           2.55           2.82           1.76
surface_pressure                   uint32             1.72           1.96           3.17           3.15           3.32           2.69           3.87           4.30           2.71
mixing_ratio                       uint16             1.11           1.14           1.53           1.53           1.60           1.27           1.68           1.82           1.33
specific_humidity                  uint16             1.11           1.15           1.53           1.54           1.60           1.28           1.69           1.83           1.33
dew_point_temperature              uint16             1.33           1.58           1.87           1.87           2.40           1.69           2.50           2.77           1.73
precipitation                      uint16             3.13           3.24           3.84           3.84           3.93           2.74           3.30           3.85           1.98
wind_speed                         uint16             1.31           1.47           1.80           1.80           2.05           1.56           2.23           2.40           1.63
wind_direction                     uint16             1.23           1.36           1.77           1.77           2.01           1.54           2.13           2.33           1.51
soil_temperature                   uint16             1.36           1.64           1.87           1.86           2.40           1.70           2.40           2.65           1.88
shortwave_radiation                uint16             2.05           2.11           2.57           2.58           2.82           2.14           2.47           2.68           1.14
longwave_radiation                 uint16             1.31           1.45           1.70           1.70           1.88           1.45           1.98           2.08           1.22
sensible_heat_flux                 uint16             1.44           1.66           1.83           1.84           2.24           1.65           2.22           2.46           1.67
moisture_flux                      uint16             2.57           2.89           2.48           2.58           4.30           2.07           3.95           4.50           2.17
potential_temperature              uint16             1.36           1.65           1.89           1.89           2.54           1.75           2.55           2.84           1.76
equivalent_potential_temperature   uint16             1.19           1.28           1.75           1.76           1.93           1.50           2.07           2.30           1.55
u_wind                             uint16             1.27           1.40           1.80           1.80           2.04           1.56           2.22           2.40           1.63
v_wind                             uint16             1.27           1.41           1.79           1.79           2.04           1.55           2.26           2.45           1.63
vorticity                          uint16             4.11           4.21           5.16           5.09           5.57           2.75           5.54           6.19           2.29
mslp                               uint32             1.73           1.99           3.22           3.21           3.41           2.74           3.97           4.49           2.73
pwat                               uint16             1.77           2.13           1.96           1.99           3.32           1.84           3.49           4.03           2.09
vimf_u                             uint32             2.07           2.27           3.30           3.30           3.50           2.79           4.09           4.58           2.90
vimf_v                             uint32             2.04           2.24           3.27           3.28           3.42           2.74           3.87           4.24           2.92
ivt                                float32            1.13           1.13           1.32           1.32           1.50           1.28           1.52           1.58           1.35

=== per-variable compress MB/s (hard vars) ===
var                                dtype    zstd-1 (curren         zstd-3 npshuffle+zstd blosc2 zstd-1  blosc2 zstd-1  blosc2 lz4 shu      pcodec L4 pcodec L8 (def   zfp lossless
air_temperature                    uint16              689            217           1373           1713           1160           3150            250            213            164
surface_pressure                   uint32              496            252           1313           1528           1356           2710            459            364            170
mixing_ratio                       uint16             1270            674            761            786            775           1501            238            169            163
specific_humidity                  uint16             1278            661            764            796            780           1536            237            169            164
dew_point_temperature              uint16              882            220           1342           1562           1102           2805            248            209            163
precipitation                      uint16              621            387            826            909            873           1852            284            252            167
wind_speed                         uint16             1210            231           1165           1191            877           2001            239            199            162
wind_direction                     uint16             1133            243           1069           1205            879           2039            238            196            164
soil_temperature                   uint16              868            221           1259           1571           1106           2937            248            211            163
shortwave_radiation                uint16             1649            602           1100           1362           1076           2600            269            212            165
longwave_radiation                 uint16             1079            241            912            994            782           1887            238            195            164
sensible_heat_flux                 uint16              715            231           1199           1338           1014           2674            244            208            164
moisture_flux                      uint16              428            306           1012           1100            851           3194            279            263            165
potential_temperature              uint16              760            217           1424           1724           1179           3254            249            214            164
equivalent_potential_temperature   uint16             1242            259           1046           1097            826           1960            239            194            164
u_wind                             uint16             1207            237           1176           1247            894           2180            241            200            164
v_wind                             uint16             1208            236           1152           1197            880           2113            241            200            163
vorticity                          uint16              557            464            748            794            776           1563            313            286            165
mslp                               uint32              486            251           1313           1555           1402           2640            459            366            171
pwat                               uint16              485            237           1412           1516            867           3597            261            245            165
vimf_u                             uint32              439            281           1373           1585           1325           2651            464            376            171
vimf_v                             uint32              452            286           1362           1648           1398           2844            458            363            171
ivt                                float32            1265           1035           1760           1840           1203           3611            418            279            339

=== per-variable decompress MB/s (hard vars) ===
var                                dtype    zstd-1 (curren         zstd-3 npshuffle+zstd blosc2 zstd-1  blosc2 zstd-1  blosc2 lz4 shu      pcodec L4 pcodec L8 (def   zfp lossless
air_temperature                    uint16             1655           1054            758           4218           2039           6469           1020           1013            360
surface_pressure                   uint32             1752           1442           1164           3752           2441           7494           1991           1976            534
mixing_ratio                       uint16             2008           1714            668           2296           1898           4551           1013           1008            297
specific_humidity                  uint16             2019           1703            668           2326           1891           4557           1017           1007            300
dew_point_temperature              uint16             1737           1089            748           3759           1878           5804           1015           1009            354
precipitation                      uint16             1850           1675            663           2265           1984           5209           1196           1111            395
wind_speed                         uint16             1883           1183            706           2910           1517           4594           1013           1009            338
wind_direction                     uint16             1861           1212            706           3042           1528           4771           1015           1009            321
soil_temperature                   uint16             1634           1068            752           3880           1867           6321           1056           1034            376
shortwave_radiation                uint16             3034           2503            757           4327           2331           8317           1065           1050            278
longwave_radiation                 uint16             1807           1189            679           2439           1408           4226           1115           1108            290
sensible_heat_flux                 uint16             2054           1133            733           3026           1758           5772           1038           1065            348
moisture_flux                      uint16             1601           1487            694           2942           2480          10518           1049           1062            413
potential_temperature              uint16             1642           1061            761           4267           2065           6607           1023           1011            360
equivalent_potential_temperature   uint16             1892           1275            689           2617           1395           4416           1015           1008            331
u_wind                             uint16             1876           1183            710           2941           1514           4705           1013           1008            338
v_wind                             uint16             1882           1201            709           2877           1488           4656           1014           1009            336
vorticity                          uint16             1457           1417            689           2514           2185           5142           1116           1109            429
mslp                               uint32             1729           1430           1170           3971           2507           7659           1824           1804            548
pwat                               uint16             1561           1217            789           3997           2139           8449           1013           1009            394
vimf_u                             uint32             1670           1493           1172           3568           2236           5805           1814           1804            554
vimf_v                             uint32             1688           1490           1184           4167           2520           7075           1997           1977            559
ivt                                float32            2025           1865           1329           7404           3223          10244           1944           1947            363

## Dependency footprint
- blosc2 4.13.1: 22 MB wheel; requires httpx, msgpack, ndindex, numexpr, numpy, pydantic, rich, threadpoolctl
- python-blosc 1.11.4 (c-blosc1): no deps; has SHUFFLE/BITSHUFFLE but not BYTEDELTA
- pcodec 1.0.3: 1.5 MB wheel; requires numpy only. Supports i8..i64, u8..u64, f16/f32/f64;
  NOT bool or datetime64 (would need a view as uint8 / int64); not strings/objects/geometry.
- zstandard (current): 23 MB wheel.

## Files
extract_chunks.py, bench.py, bench_small.py, bench_float.py, analyse.py, results_all.json, *.txt (the run outputs)

## Round 2: dependency-free alternatives to blosc2 (HARD subset; ALL file = projected d01.cfdb size)
zstd-1 (current)                             HARD ratio= 1.53 size= 550.1 MB comp=  699 MB/s decomp= 1792 MB/s | ALL file= 556.3 MB
numpy transpose-shuffle + zstd-1 (prev)      HARD ratio= 2.13 size= 396.0 MB comp= 1157 MB/s decomp=  829 MB/s | ALL file= 400.6 MB
numpy bitop-shuffle + zstd-1                 HARD ratio= 2.13 size= 396.0 MB comp= 1226 MB/s decomp= 2888 MB/s | ALL file= 400.6 MB
numpy bitop-shuffle + bytedelta + zstd-1     HARD ratio= 2.47 size= 342.2 MB comp=  648 MB/s decomp=  536 MB/s | ALL file= 346.4 MB
numpy bitop-shuffle + bytedelta + zstd-3     HARD ratio= 2.47 size= 341.9 MB comp=  591 MB/s decomp=  527 MB/s | ALL file= 346.1 MB
blosc1 zstd-1 shuffle (zero-dep pkg)         HARD ratio= 2.08 size= 406.8 MB comp=  991 MB/s decomp= 3348 MB/s | ALL file= 427.0 MB
blosc2 zstd-1 shuffle                        HARD ratio= 2.14 size= 394.9 MB comp= 1292 MB/s decomp= 3372 MB/s | ALL file= 414.7 MB
blosc2 zstd-1 shuffle+bytedelta              HARD ratio= 2.47 size= 342.1 MB comp= 1025 MB/s decomp= 2006 MB/s | ALL file= 358.4 MB
shuffle, delta all planes, cumsum u1                 HARD ratio= 2.47 size= 342.2 MB comp=  950 MB/s decomp=  532 MB/s | ALL file= 346.4 MB
shuffle, delta all planes, cumsum i8->u1             HARD ratio= 2.47 size= 342.2 MB comp=  931 MB/s decomp=  695 MB/s | ALL file= 346.4 MB
shuffle, delta planes>=1 (skip low byte)             HARD ratio= 2.18 size= 387.1 MB comp= 1150 MB/s decomp=  843 MB/s | ALL file= 391.8 MB
shuffle, delta top plane only (u16: byte1; u32: byte3) HARD ratio= 2.14 size= 394.8 MB comp= 1189 MB/s decomp= 1081 MB/s | ALL file= 399.5 MB
numpy bitop-shuffle + zstd-1                           HARD ratio= 2.13 size= 396.0 MB comp= 1222 MB/s decomp= 2866 MB/s | ALL file= 400.6 MB
numpy elem-delta(y,rowloop) + bitop-shuffle + zstd-1   HARD ratio= 2.29 size= 367.9 MB comp=  795 MB/s decomp= 1230 MB/s | ALL file= 373.1 MB
numpy bitop-shuffle + plane-delta(y,rowloop) + zstd-1  HARD ratio= 2.49 size= 339.3 MB comp=  954 MB/s decomp= 1243 MB/s | ALL file= 343.3 MB
blosc2 zstd-1 shuffle+bytedelta (ref)                  HARD ratio= 2.47 size= 342.1 MB comp= 1023 MB/s decomp= 2003 MB/s | ALL file= 358.4 MB
plane-delta along time (axis 0, 24 steps)        HARD ratio= 2.15 size= 392.7 MB comp=  980 MB/s decomp= 2054 MB/s | ALL file= 397.4 MB
plane-delta along y (axis 2, 324 steps)          HARD ratio= 2.49 size= 339.3 MB comp=  946 MB/s decomp= 1254 MB/s | ALL file= 343.3 MB
plane-delta along x (axis 3, 277 steps)          HARD ratio= 2.47 size= 342.4 MB comp=  879 MB/s decomp=  893 MB/s | ALL file= 346.8 MB

## Round 2: ratio / speed vs block size (7 hard vars)
=== COMPRESSION RATIO vs block shape (7 hard vars; elems x 2-4 B = raw bytes) ===
block shape           elems           zstd-1 (current)         np shuffle+zstd-1  np shuffle+ydelta+zstd-1     blosc1 shuffle zstd-1                 pcodec L8
(24, 1, 324, 277)   2153952                       1.47                      2.08                      2.33                      2.02                      2.64
(24, 1, 162, 277)   1076976                       1.47                      2.08                      2.33                      2.02                      2.65
(24, 1, 81, 138)     268272                       1.51                      2.09                      2.34                      2.02                      2.64
(24, 1, 40, 69)       66240                       1.54                      2.05                      2.31                      2.03                      2.62
(24, 1, 20, 34)       16320                       1.54                      2.04                      2.24                      2.03                      2.57
(24, 1, 10, 17)        4080                       1.53                      2.02                      2.15                      2.01                      2.47
(24, 1, 5, 8)           960                       1.45                      1.90                      1.91                      1.87                      2.32
(1, 1, 324, 277)      89748                       1.46                      2.05                      2.28                      2.02                      2.64
(1, 1, 162, 138)      22356                       1.48                      2.03                      2.25                      2.02                      2.62
(1, 1, 81, 69)         5589                       1.53                      2.02                      2.24                      2.02                      2.58
(1, 1, 40, 34)         1360                       1.51                      1.99                      2.17                      1.96                      2.51
(1, 1, 20, 17)          340                       1.43                      1.86                      1.94                      1.78                      2.32
(1, 1, 10, 8)            80                       1.21                      1.61                      1.61                      1.37                      1.87

=== COMPRESS MB/s vs block shape (7 hard vars; elems x 2-4 B = raw bytes) ===
block shape           elems           zstd-1 (current)         np shuffle+zstd-1  np shuffle+ydelta+zstd-1     blosc1 shuffle zstd-1                 pcodec L8
(24, 1, 324, 277)   2153952                        718                      1185                       961                      1009                       248
(24, 1, 162, 277)   1076976                        720                      1348                      1082                      1027                       254
(24, 1, 81, 138)     268272                        706                      1361                      1057                      1047                       241
(24, 1, 40, 69)       66240                        671                      1130                       920                      1015                       208
(24, 1, 20, 34)       16320                        699                       980                       686                      1025                       118
(24, 1, 10, 17)        4080                        540                       638                       337                       730                        85
(24, 1, 5, 8)           960                        374                       297                       115                       356                        44
(1, 1, 324, 277)      89748                        721                      1216                       930                      1031                       215
(1, 1, 162, 138)      22356                        734                      1073                       782                       981                       135
(1, 1, 81, 69)         5589                        564                       724                       450                       809                        60
(1, 1, 40, 34)         1360                        419                       368                       172                       440                        34
(1, 1, 20, 17)          340                        205                       144                        55                       176                        19
(1, 1, 10, 8)            80                         73                        43                        14                        56                        13

=== DECOMPRESS MB/s vs block shape (7 hard vars; elems x 2-4 B = raw bytes) ===
block shape           elems           zstd-1 (current)         np shuffle+zstd-1  np shuffle+ydelta+zstd-1     blosc1 shuffle zstd-1                 pcodec L8
(24, 1, 324, 277)   2153952                       1814                      2814                      1294                      3544                      1359
(24, 1, 162, 277)   1076976                       1818                      2867                      1326                      3496                      1360
(24, 1, 81, 138)     268272                       1872                      2873                      1040                      3543                      1352
(24, 1, 40, 69)       66240                       1728                      2587                       691                      3232                      1318
(24, 1, 20, 34)       16320                       1564                      1896                       409                      2809                      1196
(24, 1, 10, 17)        4080                       1064                       994                       209                      1630                       927
(24, 1, 5, 8)           960                        507                       349                        86                       659                       499
(1, 1, 324, 277)      89748                       1776                      2616                       475                      3415                      1312
(1, 1, 162, 138)      22356                       1581                      2139                       258                      3008                      1204
(1, 1, 81, 69)         5589                       1150                      1243                       132                      2038                       933
(1, 1, 40, 34)         1360                        631                       467                        61                       863                       542
(1, 1, 20, 17)          340                        240                       152                        26                       303                       222
(1, 1, 10, 8)            80                         68                        39                        10                        83                        67

Scripts: bench_numpy*.py, bench_sizes.py; results_sizes.json
