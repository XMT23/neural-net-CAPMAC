#!/bin/bash
THREADS=${THREADS:-"2 4 6 12"}
REPS=${REPS:-3}
CONFIG=${CONFIG:-configuration/configfile.txt}
OUT=results/bench_${SLURM_JOB_ID:-local}.csv

mkdir -p results
rm -rf build
make -s PAR=0 || exit 1
make -s PAR=1 || exit 1

run() {
  out=$(OMP_NUM_THREADS=$2 $1 $CONFIG)
  t=$(echo "$out" | grep TRAIN_TIME | cut -d= -f2)
  l=$(echo "$out" | grep '^Epoch' | tail -1 | awk '{print $4}')
  echo "$t $l"
}
median() { sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }

# Sequential - no OMP
ts=(); for r in $(seq $REPS); do read t l < <(run build/release-seq/neural_net 1); ts+=($t); lseq=$l; done
T_SEQ=$(printf "%s\n" "${ts[@]}" | median)

echo "version,threads,time,speedup,efficiency,loss" > $OUT
echo "seq,1,$T_SEQ,1.00,1.00,$lseq" >> $OUT

for n in $THREADS; do
  ts=(); for r in $(seq $REPS); do read t l < <(run build/release-par/neural_net $n); ts+=($t); lp=$l; done
  T=$(printf "%s\n" "${ts[@]}" | median)
  echo "par,$n,$T,$(echo "$T_SEQ $T $n" | awk '{printf "%.2f,%.2f", $1/$2, $1/$2/$3}'),$lp" >> $OUT
done

column -s, -t $OUT
