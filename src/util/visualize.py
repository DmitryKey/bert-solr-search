from utils import fbin_to_tsv

TOTAL_ELEMS = 500000
fbin_to_tsv("data/big_ann/yandex/text2image-1b/query.learn.50M.fbin", "data/viz/query.learn.50M.fbin_" + str(TOTAL_ELEMS) + ".tsv", TOTAL_ELEMS)
