from utils import fbin_to_tsv

TOTAL_ELEMS = 1000000
fbin_to_tsv("data/big_ann/yandex/text2image-1b/query.learn.50M.fbin", "data/viz/query.learn.50M.fbin" + str(TOTAL_ELEMS) + ".tsv", TOTAL_ELEMS)
