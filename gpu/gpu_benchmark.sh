# script to automatically benchmark the performance of the deblurring algos
# for each dataset, call deblur

echo processing beach
./gpu_deblur.out ../data/beach_blurry.png ./output/benchmarks/beach_deblur ../data/beach.png > ./output/benchmarks/beach.log

echo processing chrys
./gpu_deblur.out ../data/chrys_blurry.png ./output/benchmarks/chrys_deblur ../data/chrys.png > ./output/benchmarks/chrys.log

echo processing dc
./gpu_deblur.out ../data/dc_blurry.png ./output/benchmarks/dc_deblur ../data/dc.png > ./output/benchmarks/dc.log

echo processing desert
./gpu_deblur.out ../data/desert_blurry.png ./output/benchmarks/desert_deblur ../data/desert.png > ./output/benchmarks/desert.log

echo processing homes
./gpu_deblur.out ../data/homes_blurry.png ./output/benchmarks/homes_deblur ../data/homes.png > ./output/benchmarks/homes.log

echo processing hydra
./gpu_deblur.out ../data/hydra_blurry.png ./output/benchmarks/hydra_deblur ../data/hydra.png > ./output/benchmarks/hydra.log

echo processing jellyfish
./gpu_deblur.out ../data/jellyfish_blurry.png ./output/benchmarks/jellyfish_deblur ../data/jellyfish.png > ./output/benchmarks/jellyfish.log

echo processing koala
./gpu_deblur.out ../data/koala_blurry.png ./output/benchmarks/koala_deblur ../data/koala.png > ./output/benchmarks/koala.log

echo processing lighthouse
./gpu_deblur.out ../data/lighthouse_blurry.png ./output/benchmarks/lighthouse_deblur ../data/lighthouse.png > ./output/benchmarks/lighthouse.log

echo processing mandrill
./gpu_deblur.out ../data/mandrill_blurry.png ./output/benchmarks/mandrill_deblur ../data/mandrill.png > ./output/benchmarks/mandrill.log

echo processing mount
./gpu_deblur.out ../data/mount_blurry.png ./output/benchmarks/mount_deblur ../data/mount.png > ./output/benchmarks/mount.log

echo processing nasa
./gpu_deblur.out ../data/nasa_earth_blurry.png ./output/benchmarks/nasa_earth_deblur ../data/nasa_earth.png > ./output/benchmarks/nasa_earth.log

echo processing penguins
./gpu_deblur.out ../data/penguins_blurry.png ./output/benchmarks/penguins_deblur ../data/penguins.png > ./output/benchmarks/penguins.log

echo processing peppers
./gpu_deblur.out ../data/peppers_blurry.png ./output/benchmarks/peppers_deblur ../data/peppers.png > ./output/benchmarks/peppers.log

echo processing star
./gpu_deblur.out ../data/star_blurry.png ./output/benchmarks/star_deblur ../data/star.png > ./output/benchmarks/star.log

echo processing tulips
./gpu_deblur.out ../data/tulips_blurry.png ./output/benchmarks/tulips_deblur ../data/tulips.png > ./output/benchmarks/tulips.log

cat ./output/benchmarks/*.log > ./output/benchmarks/masterlog.log
echo done!
