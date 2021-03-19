#!/bin/bash
#SBATCH --parsable
#SBATCH --open-mode=truncate
#SBATCH -t 1
#SBATCH -p priority
#SBATCH --mem=1G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu

#Define arguments
args=( 
	--lmm
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr{1:22}_v2.bed
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr{1:22}_v2.bim
	--fam=/n/groups/patel/Alan/Aging/Medical_Images/data/GWAS.fam
	--phenoFile=/n/groups/patel/Alan/Aging/Medical_Images/data4/GWAS_exhaustive_placeholder.tab
	--phenoCol=phenotype
	--LDscoresFile=/n/groups/patel/bin/BOLT-LMM_v2.3.2/tables/LDSCORE.1000G_EUR.tab.gz
	--geneticMapFile=/n/groups/patel/bin/BOLT-LMM_v2.3.2/tables/genetic_map_hg19.txt.gz
	--numThreads=10
	--statsFile=GWAS_nonimputed.stats.gz
	--verboseStats
	--bgenSampleFileList=/n/groups/patel/uk_biobank/project_52887_genetics/bgenSampleFileList_$1.txt
	--bgenMinMAF=1e-3
	--bgenMinINFO=0.3
	--noBgenIDcheck
	--statsFileBgenSnps=/n/groups/patel/Alan/Aging/Medical_Images/data4/GWAS_nonimputed.bgen.stats.gz
)

# Add chromosomes args
if [ $1 == "X" ]; then
	args+=(
		--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chrX_v2.bed
		--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chrX_v2.bim
	)
fi

# Run the job
cd /n/groups/patel/bin/BOLT-LMM_v2.3.2/

#./bolt "${args[@]}"
output=$(./bolt "${args[@]}" 2>&1)

echo $output

# Move the non imputed ids file to data
toremove_ids=$(echo $output | grep -o "bolt.in_plink_but_not_imputed.FID_IID.*.txt")
mv $toremove_ids /n/groups/patel/Alan/Aging/Medical_Images/data/bolt.in_plink_but_not_imputed.FID_IID.$1.txt
echo "Done"

