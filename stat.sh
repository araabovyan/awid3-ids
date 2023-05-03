DATA_DIR="/data/security/CSV"

dirs=( $(ls -1 $DATA_DIR | sort) )

for d in "${dirs[@]}" ; do
    total=$(find "$DATA_DIR/$d" -type f -exec wc -l {} + | awk '{total += $1} END {print total}')
    printf "%s: %d\n" "$d" "$total"
done