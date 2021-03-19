from MI_Classes import GWASAnnotate

# Default parameters
target = 'Age'

# Compute results
GWAS_Annotate = GWASAnnotate(target=target)
GWAS_Annotate.download_data()
GWAS_Annotate.preprocessing_rs()
# Step not in python
print('Upload the snps_rs.txt file into https://www.snp-nexus.org/v4/ to link the SNPs to the genes. \n Then save the '
      'output file as GWAS_genes_rs.txt under ../data/')
GWAS_Annotate.postprocessing_rs()
GWAS_Annotate.preprocessing_chrbp()
# Step not in python
print('Upload the snps_chrbp.txt file into https://www.snp-nexus.org/v4/ to link the SNPs to the genes. \n Then save '
      'the output file as GWAS_genes_chrbp.txt under ../data/')
GWAS_Annotate.postprocessing_chrbp()
GWAS_Annotate.preprocessing_missing()
# /!\ NEEDS TO BE DONE /!\ research the missing SNPs manually:
# Submit All_hits_missing.csv at https://my.locuszoom.org/gwas/upload/
GWAS_Annotate.postprocessing_missing()
GWAS_Annotate.postprocessing_hits()
GWAS_Annotate.upload_data()
GWAS_Annotate.summarize_results()

# Exit
print('Done.')
sys.exit(0)
