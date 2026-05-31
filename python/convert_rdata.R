# convert_rdata.R
#
# Export the R `mcnp-dataset.RData` to portable files the Python/torch port can
# read. Run this once with R installed:
#
#   Rscript convert_rdata.R /path/to/extdata mcnp
#
# It writes `<code>-dataset.pkl`-equivalent CSVs that tabulate() can consume,
# plus the raw output table so the Python split/scaling can be reproduced.

args <- commandArgs(trailingOnly = TRUE)
ext.dir <- ifelse(length(args) >= 1, args[1], '.')
code <- ifelse(length(args) >= 2, args[2], 'mcnp')

rdata <- file.path(ext.dir, paste0(code, '-dataset.RData'))
load(rdata) # loads `dataset`

# The simplest portable export: write the raw MCNP output table. The Python
# tabulate() rebuilds the train/test split and scaling from this CSV.
utils::write.csv(dataset$output, file.path(ext.dir, paste0(code, '.csv')), row.names = FALSE)

# Also export the pre-split/scaled frames for exact reproduction if desired.
utils::write.csv(dataset$training.data, file.path(ext.dir, paste0(code, '-training-data.csv')), row.names = FALSE)
utils::write.csv(dataset$test.data, file.path(ext.dir, paste0(code, '-test-data.csv')), row.names = FALSE)
utils::write.csv(as.data.frame(dataset$training.df), file.path(ext.dir, paste0(code, '-training-df.csv')), row.names = FALSE)
utils::write.csv(as.data.frame(dataset$test.df), file.path(ext.dir, paste0(code, '-test-df.csv')), row.names = FALSE)
utils::write.csv(data.frame(label = names(dataset$training.mean),
                            mean = as.numeric(dataset$training.mean),
                            sd = as.numeric(dataset$training.sd)),
                 file.path(ext.dir, paste0(code, '-scale-stats.csv')), row.names = FALSE)

cat('Exported', code, 'dataset to CSV in', ext.dir, '\n')
