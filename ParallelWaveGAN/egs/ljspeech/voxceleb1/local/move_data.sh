source_dir=$1
target_dir=$2
mkdir -p ${target_dir}
for spk in `ls $source_dir`;do
	for subdir in `ls ${source_dir}/${spk}`;do
		for wav in `ls ${source_dir}/${spk}/${subdir}`; do
			file="${source_dir}/${spk}/${subdir}/${wav}"
			ln -s $file ${target_dir}/${spk}_${subdir}_${wav}
		done
	done
done

