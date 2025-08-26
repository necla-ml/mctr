#!/bin/tcsh -f

set exp_dir = $1
set setting = $2

mkdir -p results/$setting

cd ${exp_dir}/checkpoints/
set files = `ls *epoch*`
cd -

echo $files

foreach file ( $files ) 
  if ( ! -e results/$setting/results_${setting}_$file ) then 
     python scripts/trackeval_mmptrack_reset1.py ${exp_dir}  $file > results/$setting/results_${setting}_$file
  endif 
end

