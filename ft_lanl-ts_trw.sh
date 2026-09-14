python pretrain.py --dirty --trw --device 3 --log-out pretrained/temporal/lanl14argus-dirtyts/

python lp_finetune.py --argus-ts --walk-len 2 --device 3 --trw
python lp_finetune.py --argus-ts --walk-len 4 --device 3 --trw
python lp_finetune.py --argus-ts --walk-len 6 --device 3 --trw
python lp_finetune.py --argus-ts --walk-len 8 --device 3 --trw
python lp_finetune.py --argus-ts --walk-len 10 --device 3 --trw
python lp_finetune.py --argus-ts --walk-len 16 --device 3 --trw
python lp_finetune.py --argus-ts --walk-len 32 --device 3 --trw

python lp_finetune.py --argus-ts --walk-len 2 --device 3 --best --trw
python lp_finetune.py --argus-ts --walk-len 4 --device 3 --best --trw
python lp_finetune.py --argus-ts --walk-len 6 --device 3 --best --trw
python lp_finetune.py --argus-ts --walk-len 8 --device 3 --best --trw
python lp_finetune.py --argus-ts --walk-len 10 --device 3 --best --trw
python lp_finetune.py --argus-ts --walk-len 16 --device 3 --best --trw
python lp_finetune.py --argus-ts --walk-len 32 --device 3 --best --trw