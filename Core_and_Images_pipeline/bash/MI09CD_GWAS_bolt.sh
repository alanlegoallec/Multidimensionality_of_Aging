#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alanlegoallec@g.edu

#Define parameters
target="${1}"
organ="${2}"
chromosomes="${3}"
analysis="${4}"

# Define number of chromosomes
if [ $# -eq 5 ] && [ $5 == "debug_mode" ]; then
	debug="_debug"
	NC=2
else
	debug=""
	NC=22
fi

# Define parameters
if [ "${analysis}" == "lmm" ]; then
	args=( --lmm )
elif [ "${analysis}" == "reml" ] || [ "${analysis}" == "reml_correlation" ]; then
	args=( --reml )
	if [ "${debug}" == "_debug" ]; then
		args+=( --remlNoRefine )
	fi
	if [ "${analysis}" == "reml_correlation" ]; then
		# The variable for organ2 is held inside $3, which is $chromosomes
		organ2="${chromosomes}"
		chromosomes="X"
	fi
fi

args+=( 
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr{1:"${NC}"}_v2.bed
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr{1:"${NC}"}_v2.bim
	--fam=/n/groups/patel/Alan/Aging/Medical_Images/data/GWAS.fam
	--remove=/n/groups/patel/Alan/Aging/Medical_Images/data/bolt.in_plink_but_not_imputed.FID_IID."${chromosomes}".txt 
	--remove=/n/groups/patel/Alan/Aging/Medical_Images/data/GWAS_remove_"${target}"_"${organ}".tab
	--phenoCol="${organ}"
	--covarCol=Assessment_center
	--covarCol=Sex
	--covarCol=Ethnicity
	--covarMaxLevels=30
	--qCovarCol=Age
	--qCovarCol=PC{1:20}
	--LDscoresFile=/n/groups/patel/bin/BOLT-LMM_v2.3.2/tables/LDSCORE.1000G_EUR.tab.gz
	--geneticMapFile=/n/groups/patel/bin/BOLT-LMM_v2.3.2/tables/genetic_map_hg19_withX.txt.gz
	--numThreads=10
	--statsFile=/n/groups/patel/Alan/Aging/Medical_Images/data/GWAS_"${target}"_"${organ}"_"${chromosomes}""${debug}".stats.gz
	--verboseStats
)

if [ $analysis == "reml_correlation" ]; then
	args+=(
		--phenoFile=/n/groups/patel/Alan/Aging/Medical_Images/data/GWAS_data_"${target}"_"${organ}"_"${organ2}".tab
		--covarFile=/n/groups/patel/Alan/Aging/Medical_Images/data/GWAS_data_"${target}"_"${organ}"_"${organ2}".tab
		--phenoCol="${organ2}"
		--remove=/n/groups/patel/Alan/Aging/Medical_Images/data/GWAS_remove_"${target}"_"${organ}"_"${organ2}".tab
	)
else
	args+=(
		--phenoFile=/n/groups/patel/Alan/Aging/Medical_Images/data/GWAS_data_"${target}"_"${organ}".tab
		--covarFile=/n/groups/patel/Alan/Aging/Medical_Images/data/GWAS_data_"${target}"_"${organ}".tab
		--remove=/n/groups/patel/Alan/Aging/Medical_Images/data/GWAS_remove_"${target}"_"${organ}".tab
	)
fi

# Add chromosome X. Still need all other chromosomes included for the optimal LMM
if [ $chromosomes == "X" ]; then
	args+=(
		--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chrX_v2.bed
		--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chrX_v2.bim
	)
fi

# Add arguments for imputed data GWAS
if [ $debug == "" ]; then
	args+=(
		--bgenSampleFileList=/n/groups/patel/uk_biobank/project_52887_genetics/bgenSampleFileList_"${chromosomes}".txt
		--bgenMinMAF=1e-3
		--bgenMinINFO=0.3
		--noBgenIDcheck
		--statsFileBgenSnps=/n/groups/patel/Alan/Aging/Medical_Images/data/GWAS_"${target}"_"${organ}"_"${chromosomes}""${debug}".bgen.stats.gz 
	)
fi

# Run the job
cd /n/groups/patel/bin/BOLT-LMM_v2.3.2/
./bolt "${args[@]}"

#Unzip the output files
if [ "${analysis}" == "lmm" ]; then
	gunzip /n/groups/patel/Alan/Aging/Medical_Images/data/GWAS_"${target}"_"${organ}"_"${chromosomes}""${debug}".stats.gz
	if [ "${debug}" == "" ]; then
		gunzip /n/groups/patel/Alan/Aging/Medical_Images/data/GWAS_"${target}"_"${organ}"_"${chromosomes}""${debug}".bgen.stats.gz
	fi
fi

echo "Done"

